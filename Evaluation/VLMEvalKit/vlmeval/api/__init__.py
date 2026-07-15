from .gpt import OpenAIWrapper, GPT4V
from .hf_chat_model import HFChatModel


class _UnavailableVendoredAPI:
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "This API adapter was omitted from the vendored VLMEvalKit snapshot. "
            "It is not required by the local CoLT evaluation profile."
        )


GeminiWrapper = Gemini = _UnavailableVendoredAPI
QwenVLWrapper = QwenVLAPI = Qwen2VLAPI = QwenAPI = _UnavailableVendoredAPI
from .claude import Claude_Wrapper, Claude3V
Reka = _UnavailableVendoredAPI
from .glm_vision import GLMVisionAPI
from .cloudwalk import CWWrapper
SenseChatVisionAPI = _UnavailableVendoredAPI
from .siliconflow import SiliconFlowAPI, TeleMMAPI
from .hunyuan import HunyuanVision
bailingMMAPI = BlueLMWrapper = BlueLM_API = _UnavailableVendoredAPI
from .jt_vl_chat import JTVLChatAPI
from .jt_vl_chat_mini import JTVLChatAPI_Mini
from .taiyi import TaiyiAPI
from .lmdeploy import LMDeployAPI
from .taichu import TaichuVLAPI, TaichuVLRAPI
from .doubao_vl_api import DoubaoVL
from .mug_u import MUGUAPI
from .kimivl_api import KimiVLAPIWrapper, KimiVLAPI

__all__ = [
    'OpenAIWrapper', 'HFChatModel', 'GeminiWrapper', 'GPT4V', 'Gemini',
    'QwenVLWrapper', 'QwenVLAPI', 'QwenAPI', 'Claude3V', 'Claude_Wrapper',
    'Reka', 'GLMVisionAPI', 'CWWrapper', 'SenseChatVisionAPI', 'HunyuanVision',
    'Qwen2VLAPI', 'BlueLMWrapper', 'BlueLM_API', 'JTVLChatAPI', 'JTVLChatAPI_Mini',
    'bailingMMAPI', 'TaiyiAPI', 'TeleMMAPI', 'SiliconFlowAPI', 'LMDeployAPI',
    'TaichuVLAPI', 'TaichuVLRAPI', 'DoubaoVL', "MUGUAPI", 'KimiVLAPIWrapper', 'KimiVLAPI'
]
