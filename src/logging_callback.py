from transformers.integrations import WandbCallback


class LoggingCallback(WandbCallback):
    """
    Overrigding WandbCallback to optionally turn off gradient logging.
    """

    def __init__(self, log_grads: bool):

        super().__init__()

        self.log_grads = log_grads

    def setup(self, args, state, model, **kwargs):

        super().setup(args, state, model, **kwargs)
        _watch_model = "all" if self.log_grads else "parameters"
        self._wandb.watch(
            model, log=_watch_model, log_freq=max(100, args.logging_steps)
        )
