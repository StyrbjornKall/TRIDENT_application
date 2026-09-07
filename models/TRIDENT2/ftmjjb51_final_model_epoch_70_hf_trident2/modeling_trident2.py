from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedModel

from .configuration_trident2 import TRIDENT2Config


class TaxonomicEmbedder(nn.Module):
    def __init__(
        self,
        taxid_to_index: Dict[str, int],
        embedding_dim: int = 768,
        num_taxa: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.taxid_to_index = taxid_to_index or {}
        self.embedding_dim = embedding_dim
        self.taxonomic_table = nn.Embedding(num_taxa, embedding_dim)
        self.external_proj = nn.Linear(embedding_dim, embedding_dim)
        self.external_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _clean_taxid(taxid: str) -> str:
        return taxid.split("_")[0].split(",")[0].split(".")[0].strip()

    def _lookup_single(self, taxids: List[Optional[str]]) -> torch.Tensor:
        device = self.taxonomic_table.weight.device
        embeddings = []
        for taxid in taxids:
            if taxid is None:
                embeddings.append(torch.zeros(self.embedding_dim, device=device))
                continue
            clean = self._clean_taxid(str(taxid))
            idx = self.taxid_to_index.get(clean)
            if idx is None:
                embeddings.append(torch.zeros(self.embedding_dim, device=device))
            else:
                embeddings.append(
                    self.taxonomic_table(
                        torch.tensor(idx, dtype=torch.long, device=device)
                    )
                )
        return torch.stack(embeddings, dim=0).unsqueeze(1)

    def _lookup_multi(self, taxids: List[List[str]]) -> torch.Tensor:
        device = self.taxonomic_table.weight.device
        rows = []
        for taxid_list in taxids:
            vecs = []
            for t in taxid_list:
                idx = self.taxid_to_index.get(self._clean_taxid(str(t)))
                if idx is None:
                    vecs.append(torch.zeros(self.embedding_dim, device=device))
                else:
                    vecs.append(
                        self.taxonomic_table(
                            torch.tensor(idx, dtype=torch.long, device=device)
                        )
                    )
            rows.append(torch.stack(vecs, dim=0))
        return torch.stack(rows, dim=0)

    def forward(
        self, taxids: Union[List[Optional[str]], List[List[str]]]
    ) -> torch.Tensor:
        if len(taxids) == 0:
            return torch.zeros(
                (0, 1, self.embedding_dim), device=self.taxonomic_table.weight.device
            )

        if taxids[0] is None or isinstance(taxids[0], str):
            embeddings = self._lookup_single(taxids)
        else:
            embeddings = self._lookup_multi(taxids)

        embeddings = self.external_proj(embeddings)
        embeddings = self.external_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MultimodalRoBERTa(nn.Module):
    def __init__(
        self,
        base_model,
        padding_idx: int = 1,
        prepend_external: bool = True,
        taxid_to_index: Optional[Dict[str, int]] = None,
        embedding_dim: int = 768,
        num_taxa: int = 1,
    ):
        super().__init__()
        self.base_model = base_model
        self.base_model_type = base_model.config.model_type
        self.embedding_dim = base_model.embeddings.word_embeddings.embedding_dim
        self.padding_idx = padding_idx
        self.prepend_external = prepend_external
        self.has_taxonomic_embedder = (
            taxid_to_index is not None and len(taxid_to_index) > 0 and num_taxa > 1
        )
        if self.has_taxonomic_embedder:
            self.taxonomic_embedder = TaxonomicEmbedder(
                taxid_to_index=taxid_to_index,
                embedding_dim=embedding_dim,
                num_taxa=num_taxa,
                dropout=0.1,
            )

    def create_position_ids_from_input_ids(self, input_ids, padding_idx):
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
        return incremental_indices.long() + padding_idx

    def combine_embeddings(
        self,
        input_embeddings,
        input_attention_mask,
        input_position_ids,
        external_embeds,
    ):
        if external_embeds.dim() == 2:
            external_embeds = external_embeds.unsqueeze(1)

        external_mask = torch.ones(
            (input_attention_mask.size(0), 1),
            dtype=input_attention_mask.dtype,
            device=input_attention_mask.device,
        )
        # Position 0 is never used by RoBERTa under normal operation (real tokens
        # start at padding_idx+1, padding uses padding_idx). Assigning 0 here gives
        # the taxonomic token its own dedicated, learnable position embedding without
        # colliding with padding tokens, while still being position-agnostic in intent.
        external_position_ids = torch.zeros(
            (input_position_ids.size(0), 1),
            dtype=input_position_ids.dtype,
            device=input_position_ids.device,
        )

        if self.prepend_external:
            # Insert tax token at position 1 (after CLS), not at position 0
            combined_input_embeddings = torch.cat(
                [
                    input_embeddings[:, :1, :],
                    external_embeds,
                    input_embeddings[:, 1:, :],
                ],
                dim=1,
            )
            combined_attention_mask = torch.cat(
                [
                    input_attention_mask[:, :1],
                    external_mask,
                    input_attention_mask[:, 1:],
                ],
                dim=1,
            )
            combined_position_ids = torch.cat(
                [
                    input_position_ids[:, :1],
                    external_position_ids,
                    input_position_ids[:, 1:],
                ],
                dim=1,
            )
        else:
            combined_input_embeddings = torch.cat(
                [input_embeddings, external_embeds], dim=1
            )
            combined_attention_mask = torch.cat(
                [input_attention_mask, external_mask], dim=1
            )
            combined_position_ids = torch.cat(
                [input_position_ids, external_position_ids], dim=1
            )

        return combined_input_embeddings, combined_attention_mask, combined_position_ids

    def forward(self, input_ids, attention_mask, taxids=None):
        input_embeddings = self.base_model.embeddings.word_embeddings(input_ids)
        position_ids = self.create_position_ids_from_input_ids(
            input_ids=input_ids,
            padding_idx=self.padding_idx,
        )

        if taxids is not None and self.has_taxonomic_embedder:
            external_embeds = self.taxonomic_embedder(taxids=taxids)
            input_embeddings, attention_mask, position_ids = self.combine_embeddings(
                input_embeddings=input_embeddings,
                input_attention_mask=attention_mask,
                input_position_ids=position_ids,
                external_embeds=external_embeds,
            )

        outputs = self.base_model(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs


class RegressionModule(nn.Module):
    def __init__(
        self,
        input_layer_size: int,
        hidden_layer_sizes: Optional[List[int]],
        dropout: float,
        activation: str = "GELU",
        n_outputs: int = 1,
    ):
        super().__init__()
        self.hidden_layer_sizes = hidden_layer_sizes or []
        self.activation_fn = getattr(nn, activation, nn.GELU)()
        self.hidden_layers = nn.ModuleList()

        current_size = input_layer_size
        for size in self.hidden_layer_sizes:
            self.hidden_layers.append(nn.Linear(current_size, size))
            current_size = size

        self.out_proj = nn.Linear(current_size, n_outputs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation_fn(x)
            x = self.dropout(x)
        return self.out_proj(x).squeeze(1)


class MultiTaskRegressionModule(nn.Module):
    def __init__(
        self,
        input_layer_size: int,
        regression_task_configs: Dict[str, Dict],
        fusion_network_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.tasks = nn.ModuleDict()
        self.fusion = False
        self.input_layer_size = input_layer_size

        if fusion_network_config is not None:
            self.fusion_layers = RegressionModule(
                input_layer_size=input_layer_size,
                hidden_layer_sizes=fusion_network_config["hidden_layer_sizes"],
                dropout=fusion_network_config["dropout"],
                activation=fusion_network_config.get("activation", "ReLU"),
                n_outputs=fusion_network_config["n_outputs"],
            )
            self.fusion = True

        for task_name, task_cfg in regression_task_configs.items():
            task_input_size = (
                fusion_network_config["n_outputs"]
                if fusion_network_config is not None
                else input_layer_size
            )
            self.tasks[task_name] = RegressionModule(
                input_layer_size=task_input_size,
                hidden_layer_sizes=task_cfg["hidden_layer_sizes"],
                dropout=task_cfg["dropout"],
                activation=task_cfg.get("activation", "ReLU"),
                n_outputs=task_cfg["n_outputs"],
            )

    def forward(self, inputs):
        x = self.fusion_layers(inputs) if self.fusion else inputs
        return {
            task_name: task_layers(x) for task_name, task_layers in self.tasks.items()
        }


class TRIDENT2Model(PreTrainedModel):
    config_class = TRIDENT2Config
    base_model_prefix = "trident2"

    def __init__(self, config: TRIDENT2Config):
        super().__init__(config)

        if config.backbone_config is None:
            raise ValueError("backbone_config must be present in TRIDENT2 config")

        backbone_cfg = dict(config.backbone_config)
        model_type = backbone_cfg.pop("model_type")
        transformer_config = AutoConfig.for_model(model_type, **backbone_cfg)
        backbone = AutoModel.from_config(transformer_config)

        self.transformer_encoder = MultimodalRoBERTa(
            base_model=backbone,
            padding_idx=config.pad_token_id,
            prepend_external=config.prepend_external,
            taxid_to_index=config.taxid_to_index,
            embedding_dim=config.embedding_dim,
            num_taxa=config.num_taxa,
        )

        input_layer_size = (
            config.embedding_dim
            + config.onehot_length
            + (1 if config.with_duration_neuron else 0)
        )
        self.dnn = MultiTaskRegressionModule(
            input_layer_size=input_layer_size,
            regression_task_configs=config.regression_task_config,
            fusion_network_config=config.fusion_network_config,
        )

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        duration=None,
        one_hot_encoding=None,
        taxids=None,
        **kwargs,
    ):
        transformer_encoder_output = self.transformer_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            taxids=taxids,
        )[0][:, 0, :]

        tensors = [transformer_encoder_output]
        if duration is not None:
            tensors.append(duration.view(input_ids.size(0), 1))
        if one_hot_encoding is not None:
            tensors.append(one_hot_encoding)

        inputs = torch.cat(tensors, axis=1)
        out = self.dnn(inputs)
        return out, transformer_encoder_output
