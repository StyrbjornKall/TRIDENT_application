from transformers import PretrainedConfig


class TRIDENT2Config(PretrainedConfig):
    model_type = "trident2"

    def __init__(
        self,
        base_model_name_or_path="StyrbjornKall/Multimodal-ChemBERTa-1",
        backbone_config=None,
        regression_task_config=None,
        fusion_network_config=None,
        embedding_dim=768,
        onehot_length=0,
        with_duration_neuron=False,
        prepend_external=True,
        pad_token_id=1,
        taxid_to_index=None,
        num_taxa=1,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.backbone_config = backbone_config
        self.regression_task_config = regression_task_config or {}
        self.fusion_network_config = fusion_network_config
        self.embedding_dim = embedding_dim
        self.onehot_length = onehot_length
        self.with_duration_neuron = with_duration_neuron
        self.prepend_external = prepend_external
        self.taxid_to_index = taxid_to_index or {}
        self.num_taxa = num_taxa
