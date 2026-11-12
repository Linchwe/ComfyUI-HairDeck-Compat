# ComfyUI-HairDeck-Compat
#
# 용도
#  visualbruno/ComfyUI-HairDeck 노드를 아래 두 환경에서도 돌아가게 해 주는 보조 노드입니다.
#  노드 화면에는 아무것도 생기지 않고, ComfyUI 가 켜질 때 HairDeck 의 몇 군데만 바꿔 끼웁니다.
#  visualbruno 노드의 파일은 고치지 않으므로, 그 노드를 업데이트해도 이 수정은 그대로 유지됩니다.
#
# 하는 일
#  1) 계산 정밀도
#     HairDeck 과 SAM 3 는 BF16 으로 계산하는데, RTX 20 시리즈 이하(Compute Capability 7.x 이하)에는 BF16 하드웨어가 없습니다.
#     그래픽카드를 확인해서 BF16 을 지원하지 않으면 FP32 로 계산하게 바꿉니다. (느리고 VRAM 을 더 쓰지만 결과가 깨지지 않음)
#     RTX 30 시리즈 이상은 원래대로 BF16 으로 계산합니다.
#     SAM 3 이미지 인코더의 MLP 는 BF16 으로 강제 변환하는 전용 연산(addmm_act)을 쓰기 때문에, 이것도 일반 연산으로 바꿔 끼웁니다.
#  2) triton-windows 없이 SAM 3 불러오기
#     SAM 3 는 불러올 때 triton 을 import 하지만, HairDeck 이 쓰는 이미지 한 장 처리 경로에서는 triton 커널을 실행하지 않습니다.
#     triton 이 설치돼 있지 않거나 불러오다 오류가 나면, HairDeck 에 들어 있는 가짜 triton(맥용 stub)으로 SAM 3 를 불러옵니다.
#     불러온 직후 가짜 triton 은 치워서, 다른 노드가 triton 이 있다고 착각하지 않게 합니다.
#     혹시 triton 커널이 필요한 NMS 가 불리면 CPU 계산으로 대신합니다.
#  3) pkg_resources 없이 SAM 3 불러오기
#     SAM 3 는 pkg_resources 를 import 하는데, setuptools 81 이상에는 pkg_resources 가 빠져 있습니다.
#     없으면 SAM 3 가 쓰는 기능(resource_filename) 하나만 흉내 낸 모듈을 SAM 3 를 불러오는 동안만 넣고, 끝나면 치웁니다.
#     setuptools 버전은 바꾸지 않습니다.
#
# 주의
#  - visualbruno/ComfyUI-HairDeck 가 custom_nodes 안에 설치돼 있어야 합니다. 없으면 아무것도 하지 않습니다.
#  - 정밀도를 직접 고르려면 아래 PRECISION 을 바꾸거나, 환경 변수 HAIRDECK_PRECISION 에 bf16 / fp16 / fp32 를 넣으세요.
#    fp16 은 RTX 20 시리즈에서 빠르지만 값이 넘쳐 결과가 깨질 수 있어 기본값으로 쓰지 않습니다.
#  - 문제가 생기면 이 폴더(ComfyUI-HairDeck-Compat)를 지우고 ComfyUI 를 다시 켜면 원래대로 돌아갑니다.
#
# 사용법
#  ComfyUI\custom_nodes 안에 ComfyUI-HairDeck-Compat 폴더째로 넣고 ComfyUI 를 다시 켭니다.
#  콘솔(검은 창)에 [HairDeck-Compat] 로 시작하는 줄이 나오면 적용된 것입니다.

import glob
import importlib
import importlib.util
import os
import sys

PRECISION = "auto"      # auto / bf16 / fp16 / fp32   (auto: RTX 30 이상 bf16, 그 아래 fp32)

TAG = "[HairDeck-Compat]"
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _log(msg):
    print(f"{TAG} {msg}")


def _find_vendor():
    # custom_nodes 안에서 HairDeck 원본 코드(HairDeck/hairdeck/dev.py)가 들어 있는 폴더를 찾음
    here = os.path.dirname(os.path.abspath(__file__))
    custom = os.path.dirname(here)
    preferred = os.path.join(custom, "ComfyUI-HairDeck", "HairDeck")
    if os.path.isfile(os.path.join(preferred, "hairdeck", "dev.py")):
        return os.path.realpath(preferred)
    for p in glob.glob(os.path.join(custom, "*", "HairDeck", "hairdeck", "dev.py")):
        if os.path.dirname(os.path.dirname(p)) != here:
            return os.path.realpath(os.path.dirname(os.path.dirname(p)))
    return None


def _precision(torch):
    want = (os.environ.get("HAIRDECK_PRECISION") or PRECISION or "auto").strip().lower()
    if want in ("bf16", "fp16", "fp32"):
        return want, "직접 지정"
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
        if major >= 8:
            return "bf16", f"{name} (sm{major}{minor}) BF16 지원"
        return "fp32", f"{name} (sm{major}{minor}) BF16 하드웨어 없음"
    return "bf16", "CUDA 없음"


# 1) 계산 정밀도

def _patch_amp(dev, torch, prec):
    if getattr(dev.amp, "_hd_compat", False):
        return

    def amp(device, enabled=True):
        kind = torch.device(device).type
        on = bool(enabled) and kind in ("cuda", "mps")
        dtype = torch.bfloat16
        if kind == "cuda":
            if prec == "fp32":
                on = False
            elif prec == "fp16":
                dtype = torch.float16
        return torch.autocast(kind if kind in ("cuda", "mps") else "cpu", dtype=dtype, enabled=on)

    amp._hd_compat = True
    dev.amp = amp


def _safe_addmm_act(torch, orig):
    import torch.nn.functional as F

    def addmm_act(activation, linear, mat1):
        # 원래 연산은 입력과 가중치를 BF16 으로 강제 변환함. 여기서는 가중치 dtype 그대로 계산
        if activation not in (F.relu, torch.nn.ReLU, F.gelu, torch.nn.GELU):
            return orig(activation, linear, mat1)
        y = F.linear(mat1.to(linear.weight.dtype), linear.weight, linear.bias)
        return F.relu(y) if activation in (F.relu, torch.nn.ReLU) else F.gelu(y)

    addmm_act._hd_compat = True
    return addmm_act


# 2) triton 없이 SAM 3 불러오기

def _triton_names():
    return [n for n in list(sys.modules) if n == "triton" or n.startswith("triton.")]


def _real_triton_ok():
    mod = sys.modules.get("triton")
    if mod is not None:
        return not getattr(mod, "_hd_stub", False)
    try:
        if importlib.util.find_spec("triton") is None:
            return False
        importlib.import_module("triton")
        importlib.import_module("triton.language")
        return True
    except Exception:
        for n in _triton_names():
            sys.modules.pop(n, None)
        return False


def _pkg_resources_shim():
    # setuptools 81 이상에는 pkg_resources 가 없음. SAM 3 는 resource_filename 하나만 쓰므로 그것만 흉내 냄
    try:
        import pkg_resources  # noqa: F401
        return None
    except Exception:
        pass
    import types

    shim = types.ModuleType("pkg_resources")

    def resource_filename(package, resource):
        mod = sys.modules.get(package)
        if mod is not None and getattr(mod, "__file__", None):
            base = os.path.dirname(os.path.abspath(mod.__file__))
        else:
            spec = importlib.util.find_spec(package)
            base = os.path.dirname(os.path.abspath(spec.origin))
        return os.path.join(base, *resource.split("/"))

    shim.resource_filename = resource_filename
    shim._hd_shim = True
    return shim


def _patch_sam3_loader(sg, torch, prec, vendor):
    if getattr(sg._sam3, "_hd_compat", False):
        return
    orig = sg._sam3
    stub_dir = os.path.join(vendor, "hairdeck", "vendor")
    stub_mods = {}
    state = {"use_stub": None}

    def patch_sam3_modules():
        # SAM 3 모듈이 불러와진 뒤 BF16 강제 연산과 triton NMS 를 바꿔 끼움 (한 번만)
        if prec != "bf16":
            fused = sys.modules.get("sam3.perflib.fused")
            vit = sys.modules.get("sam3.model.vitdet")
            if fused is not None and not getattr(fused.addmm_act, "_hd_compat", False):
                fused.addmm_act = _safe_addmm_act(torch, fused.addmm_act)
            if vit is not None and fused is not None and not getattr(vit.addmm_act, "_hd_compat", False):
                vit.addmm_act = fused.addmm_act
        if state["use_stub"]:
            nms = sys.modules.get("sam3.perflib.nms")
            if nms is not None and not getattr(nms.generic_nms, "_hd_compat", False):
                orig_nms = nms.generic_nms

                def generic_nms(ious, scores, iou_threshold=0.5):
                    if ious.is_cuda and not getattr(nms, "GENERIC_NMS_AVAILABLE", False):
                        keep = nms.generic_nms_cpu(ious.cpu(), scores.cpu(), iou_threshold)
                        return keep.to(scores.device)
                    return orig_nms(ious, scores, iou_threshold)

                generic_nms._hd_compat = True
                nms.generic_nms = generic_nms

    def _sam3():
        # pkg_resources 가 없으면 SAM 3 를 불러오는 동안만 대신할 모듈을 넣었다가, 끝나면 치움
        shim = _pkg_resources_shim()
        if shim is None:
            return _sam3_inner()
        if not state.get("shim_logged"):
            state["shim_logged"] = True
            _log("pkg_resources: setuptools 에 없어 SAM 3 를 불러오는 동안만 대신할 모듈을 씁니다")
        sys.modules["pkg_resources"] = shim
        try:
            return _sam3_inner()
        finally:
            if sys.modules.get("pkg_resources") is shim:
                del sys.modules["pkg_resources"]

    def _sam3_inner():
        if state["use_stub"] is None:
            state["use_stub"] = not _real_triton_ok()
            _log("triton: " + ("설치돼 있지 않거나 불러올 수 없어 가짜 triton 으로 SAM 3 를 불러옵니다"
                               if state["use_stub"] else "설치된 triton 을 그대로 씁니다"))
        if not state["use_stub"]:
            result = orig()
            patch_sam3_modules()
            return result
        saved = {n: sys.modules.pop(n) for n in _triton_names()}
        sys.modules.update(stub_mods)
        added = stub_dir not in sys.path
        if added:
            sys.path.insert(0, stub_dir)
        try:
            result = orig()
            if "triton" in sys.modules:
                sys.modules["triton"]._hd_stub = True
            patch_sam3_modules()
            return result
        finally:
            if added and stub_dir in sys.path:
                sys.path.remove(stub_dir)
            for n in _triton_names():
                stub_mods[n] = sys.modules.pop(n)
            sys.modules.update(saved)

    _sam3._hd_compat = True
    sg._sam3 = _sam3


def _apply():
    vendor = _find_vendor()
    if vendor is None:
        _log("visualbruno/ComfyUI-HairDeck 를 찾지 못해 아무것도 바꾸지 않았습니다.")
        return
    try:
        import torch
    except Exception as e:
        _log(f"torch 를 불러오지 못해 건너뜁니다: {e}")
        return
    if vendor not in [os.path.realpath(p) for p in sys.path if p]:
        sys.path.insert(0, vendor)
    try:
        dev = importlib.import_module("hairdeck.dev")
        sg = importlib.import_module("hairdeck.prep.sam3_ground")
    except Exception as e:
        _log(f"HairDeck 코드를 불러오지 못해 건너뜁니다: {e}")
        return
    prec, why = _precision(torch)
    _patch_amp(dev, torch, prec)
    _patch_sam3_loader(sg, torch, prec, vendor)
    _log(f"계산 정밀도: {prec.upper()} ({why})")
    if prec == "fp32":
        _log("FP32 로 계산합니다. 속도가 느리고 VRAM 을 더 쓰니 Model Loader 의 offload 를 on 으로 두세요.")


try:
    _apply()
except Exception as e:  # 이 보조 노드 때문에 ComfyUI 가 멈추지 않게
    _log(f"적용 중 오류가 나서 건너뜁니다: {e}")
