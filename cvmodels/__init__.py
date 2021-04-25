import torch
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
        # check pretrained input channels, adapt if necessary
        for i, (k, v) in enumerate(pretrained.items()):
            k = self._layer_mapping(k)
            if k in current_state:
                # if it's input
                if i == 0:
                    pretrained_channels = v.shape[1]
                    input_channels = current_state[k].shape[1]
                    # if input channels are less than nominal 3, slice
                    if input_channels < pretrained_channels:
                        v = v[:, :pretrained_channels]
                    # if input channels are more than 3, duplicate the first N times
                    elif input_channels > pretrained_channels:
                        count = input_channels - pretrained_channels
                        additional = v[:, 0].unsqueeze(1).repeat(1, count, 1, 1)
                        v = torch.cat((v, additional), dim=1)
                updates[k] = v
        current_state.update(updates)
        self.load_state_dict(current_state)
