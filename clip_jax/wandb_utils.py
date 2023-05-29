import os
import tempfile
from contextlib import contextmanager

import wandb


class WandbMixin:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Initializes from a wandb artifact or delegates loading to the superclass.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:  # avoid multiple artifact copies
            pretrained_model_name_or_path = maybe_get_artifact(pretrained_model_name_or_path, tmp_dir)

            return super(WandbMixin, cls).from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


def maybe_get_artifact(pretrained_model_name_or_path, tmp_dir):
    if (
        ":" in pretrained_model_name_or_path
        and "gs:/" not in pretrained_model_name_or_path
        and not os.path.isdir(pretrained_model_name_or_path)
    ):
        # wandb artifact
        if wandb.run is not None:
            artifact = wandb.run.use_artifact(pretrained_model_name_or_path)
        else:
            artifact = wandb.Api().artifact(pretrained_model_name_or_path)
        return artifact.download(tmp_dir)
    else:
        return pretrained_model_name_or_path


@contextmanager
def maybe_use_artifact(file_path, artifact_file_name=None):
    try:
        if ":" in file_path and "gs:/" not in file_path:
            # wandb artifact
            if wandb.run is not None:
                artifact = wandb.run.use_artifact(file_path)
            else:
                artifact = wandb.Api().artifact(file_path)
            with tempfile.TemporaryDirectory() as tmp_dir:
                artifact_folder = artifact.download(tmp_dir)
                if artifact_file_name is not None:
                    yield os.path.join(artifact_folder, artifact_file_name)
                else:
                    yield artifact_folder
        else:
            yield file_path
    finally:
        pass
