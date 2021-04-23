from torch.nn import Module
from torch.utils import model_zoo

from cvmodels.version import __version__


def get_version():
    return __version__


class ModuleBase(Module):

    def _layer_mapping(self, layer_name: str) -> str:
        return layer_name

    def _from_pretrained(self, url: str) -> None:
        """Abstract method needed to load weights from a checkpoint
        """
        assert url is not None, "Missing pretrained URL"
        pretrained = model_zoo.load_url(url)
        updates = {}
        current_state = self.state_dict()
        for k, v in pretrained.items():
            k = self._layer_mapping(k)
            if k in current_state:
                updates[k] = v
        current_state.update(updates)
        self.load_state_dict(current_state)
