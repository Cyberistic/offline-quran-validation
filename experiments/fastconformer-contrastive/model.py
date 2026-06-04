"""FastConformer Contrastive Model for Quran verse identification.

Uses NVIDIA's Arabic FastConformer as frozen audio encoder,
adds a 256-d projection layer, and learns supervised contrastive
embeddings to match audio directly to ayah text.

Approach:
  Audio → FastConformer encoder (frozen) → projection → [256-d] → L2 normalize
  Text  → AraBERT encoder (frozen)      → projection → [256-d] → L2 normalize
  
  Retrieval: cosine similarity between audio and all ayah embeddings
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

def get_project_root():
    """Get project root for imports."""
    return Path(__file__).parent.parent.parent

sys.path.insert(0, str(get_project_root()))

PROJECT_ROOT = get_project_root()
NVIDIA_MODEL_ID = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"


class FastConformerContrastive(nn.Module):
    """
    Contrastive model aligning Quranic audio and Arabic text.
    Uses frozen FastConformer encoder + trainable projection.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        freeze_audio: bool = True,
        freeze_text: bool = True,
        text_model_name: str = "aubmindlab/bert-base-arabertv02",
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.text_model_name = text_model_name
        
        # Audio encoder: FastConformer (will be loaded separately)
        # We don't load it here to avoid import issues during model definition
        self.audio_encoder = None
        self.audio_dim = 512  # FastConformer encoder output dim
        
        # Text encoder: AraBERT
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.hidden_size  # 768
        
        # Trainable projection heads
        self.audio_proj = nn.Sequential(
            nn.Linear(self.audio_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # Learnable temperature for contrastive loss
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)
        
        # Freeze text encoder
        if freeze_text:
            self._freeze_text_encoder()
    
    def _freeze_text_encoder(self):
        """Freeze all text encoder parameters."""
        for param in self.text_encoder.parameters():
            param.requires_grad = False
    
    def set_audio_encoder(self, audio_encoder):
        """Set the FastConformer audio encoder (frozen)."""
        self.audio_encoder = audio_encoder
        # Freeze audio encoder
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        # Set to eval mode
        self.audio_encoder.eval()
    
    def encode_audio(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Encode audio features (from FastConformer) to normalized embedding.
        
        Args:
            audio_features: [batch, seq_len, 512] from FastConformer encoder
        
        Returns:
            [batch, embed_dim] L2-normalized audio embeddings
        """
        # Mean pooling over time dimension
        pooled = audio_features.mean(dim=1)  # [batch, 512]
        projected = self.audio_proj(pooled)  # [batch, embed_dim]
        return F.normalize(projected, dim=-1)
    
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode Arabic text to normalized embedding.
        
        Args:
            input_ids: [batch, seq_len] tokenized text
            attention_mask: [batch, seq_len] attention mask
        
        Returns:
            [batch, embed_dim] L2-normalized text embeddings
        """
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Use CLS token embedding
        cls_output = outputs.last_hidden_state[:, 0]  # [batch, 768]
        projected = self.text_proj(cls_output)  # [batch, embed_dim]
        return F.normalize(projected, dim=-1)
    
    def forward(
        self,
        audio_features: torch.Tensor,
        input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ):
        """
        Forward pass computing similarity logits.
        
        Returns:
            logits_per_audio: [batch, batch] audio-to-text similarity
            logits_per_text: [batch, batch] text-to-audio similarity
        """
        audio_embeds = self.encode_audio(audio_features)
        text_embeds = self.encode_text(input_ids, text_attention_mask)
        
        # Clamp temperature for stability
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        
        # Compute similarity matrix
        logits_per_audio = logit_scale * audio_embeds @ text_embeds.T
        logits_per_text = logits_per_audio.T
        
        return logits_per_audio, logits_per_text
    
    def get_trainable_params(self):
        """Return count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self):
        """Return total parameter count."""
        return sum(p.numel() for p in self.parameters())


def contrastive_loss(logits_per_audio: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    """
    Symmetric InfoNCE contrastive loss (CLIP-style).
    Each audio[i] should match text[i] (diagonal entries are positives).
    """
    batch_size = logits_per_audio.shape[0]
    labels = torch.arange(batch_size, device=logits_per_audio.device)
    loss_a = F.cross_entropy(logits_per_audio, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_a + loss_t) / 2


def get_tokenizer(model_name: str = "aubmindlab/bert-base-arabertv02"):
    """Get the AraBERT tokenizer."""
    return AutoTokenizer.from_pretrained(model_name)


class AudioFeatureExtractor:
    """
    Wrapper to extract features from frozen FastConformer encoder.
    """
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path or NVIDIA_MODEL_ID
        self._load_model()
    
    def _load_model(self):
        """Load FastConformer and extract encoder."""
        try:
            from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
            from nemo.utils import logging as nemo_logging
            nemo_logging.set_verbosity(nemo_logging.ERROR)
        except ImportError:
            raise ImportError(
                "NeMo ASR dependencies required. "
                "Install with: pip install 'nemo_toolkit[asr]'"
            )
        
        # Check for local model first
        local_dir = Path(self.model_path)
        if local_dir.exists() and (local_dir / ".nemo").exists():
            model = EncDecHybridRNNTCTCBPEModel.restore_from(str(local_dir / ".nemo"))
        else:
            model = EncDecHybridRNNTCTCBPEModel.from_pretrained(self.model_path)
        
        # Extract encoder only
        self.model = model.encoder
        self.model.eval()
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def extract_features(self, audio_batch: torch.Tensor) -> torch.Tensor:
        """
        Extract audio features from waveform.
        
        Args:
            audio_batch: [batch, samples] at 16kHz
        
        Returns:
            [batch, seq_len, 512] encoder features
        """
        return self.model(audio_batch)


if __name__ == "__main__":
    print("Testing FastConformerContrastive model...")
    
    # Test model creation
    model = FastConformerContrastive()
    print(f"Total parameters: {model.get_total_params():,}")
    print(f"Trainable parameters: {model.get_trainable_params():,}")
    
    # Test with dummy data
    batch_size = 4
    audio_features = torch.randn(batch_size, 100, 512)  # [batch, time, dim]
    input_ids = torch.randint(0, 30000, (batch_size, 32))
    text_mask = torch.ones(batch_size, 32, dtype=torch.long)
    
    print("\nRunning forward pass...")
    logits_a, logits_t = model(audio_features, input_ids, text_mask)
    print(f"Logits shape: {logits_a.shape}")
    
    loss = contrastive_loss(logits_a, logits_t)
    print(f"Loss: {loss.item():.4f}")
    
    print("\nModel test successful!")
