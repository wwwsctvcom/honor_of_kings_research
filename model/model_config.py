from transformers import PretrainedConfig


class ResNetModelConfig(PretrainedConfig):

    def __init__(
            self,
            backend_model_name="resnet101",
            classify_in_features=2048,
            action_dim=130,
            critic_dim=1,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.backend_model_name = backend_model_name
        self.classify_in_features = classify_in_features
        self.action_dim = action_dim
        self.critic_dim = critic_dim


class StateModelConfig(PretrainedConfig):

    def __init__(
            self,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=2048,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            image_size=224,
            patch_size=16,
            num_channels=3,
            qkv_bias=True,
            encoder_stride=16,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.encoder_stride = encoder_stride
