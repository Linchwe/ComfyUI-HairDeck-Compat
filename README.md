# ComfyUI-HairDeck-Compat

A helper for [visualbruno/ComfyUI-HairDeck](https://github.com/visualbruno/ComfyUI-HairDeck) that makes it run on more Windows setups.
It adds no nodes. When ComfyUI starts, it patches a few places in HairDeck at runtime; HairDeck's own files are never changed, so updating HairDeck keeps working.

What it does:

1. **Precision for RTX 20 series and older** (no bfloat16 hardware): HairDeck and SAM 3 compute in BF16. On GPUs without BF16 support it switches to FP32 (slower, more VRAM, correct results). RTX 30 series and newer stay on BF16. SAM 3's BF16-only `addmm_act` MLP op is replaced with a regular op as well.
2. **SAM 3 without triton-windows**: SAM 3 imports triton, but HairDeck's single-image path never runs a triton kernel. If triton is missing or fails to import, SAM 3 is loaded with HairDeck's bundled triton stub, which is removed again right after loading. If a triton NMS kernel is ever called, a CPU version is used.
3. **SAM 3 without `pkg_resources`**: setuptools 81+ no longer ships `pkg_resources`. Only while SAM 3 loads, a tiny module providing `resource_filename` is inserted, then removed. setuptools is not changed.

## Install

Put the `ComfyUI-HairDeck-Compat` folder into `ComfyUI/custom_nodes/` next to `ComfyUI-HairDeck` and restart ComfyUI:

```
git clone https://github.com/Linchwe/ComfyUI-HairDeck-Compat ComfyUI\custom_nodes\ComfyUI-HairDeck-Compat
```

Lines starting with `[HairDeck-Compat]` in the console mean it is active.

- Choose the precision yourself with the environment variable `HAIRDECK_PRECISION` (`bf16` / `fp16` / `fp32`) or `PRECISION` in `__init__.py`. fp16 is faster on RTX 20 series but can overflow, so it is not the default.
- If anything goes wrong, delete this folder and restart ComfyUI.

## License

MIT (see `LICENSE`). ComfyUI-HairDeck, SAM 3 and the HairDeck models keep their own licenses.

---

visualbruno/ComfyUI-HairDeck 를 더 많은 윈도우 환경(RTX 20 시리즈, triton-windows 없음, setuptools 81 이상)에서 돌아가게 해 주는 보조 노드입니다. 노드 화면에는 아무것도 생기지 않고, HairDeck 파일은 고치지 않습니다. 자세한 설명은 `__init__.py` 맨 위의 주석을 보세요.
