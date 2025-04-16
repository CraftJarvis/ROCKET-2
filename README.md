<!--
 * @Date: 2025-03-21 10:25:54
 * @LastEditors: caishaofei-mus1 1744260356@qq.com
 * @LastEditTime: 2025-03-21 10:32:18
 * @FilePath: /ROCKET2-OSS/README.md
-->
<h2 align="center"> ROCKET-2: Steering Visuomotor Policy via Cross-View Goal Alignment </h2>

<div align="center">

[`Shaofei Cai`](https://phython96.github.io/) | [`Zhancun Mu`](https://zhancunmu.owlstown.net/) | [`Anji Liu`](https://liuanji.github.io/) | [`Yitao Liang`](https://scholar.google.com/citations?user=KVzR1XEAAAAJ&hl=zh-CN&oi=ao)

All authors are affiliated with Team **[`CraftJarvis`](https://craftjarvis.github.io/)**. 

[[`Project`](https://craftjarvis.github.io/ROCKET-2/)] | [[`Paper`](https://arxiv.org/abs/2503.02505)] | [[`huggingface`]()] | [[`Demo`](https://huggingface.co/spaces/phython96/ROCKET-2-DEMO)] | [[`BibTex`](#citig_rocket)] 

</div>
<p align="center">
  <img src="images/comp_4x4_grid.gif" alt="Full Width GIF" width="100%" />
</p>

<b> ROCKET-2 was pre-trained only on Minecraft, yet generalizes zero-shot to other 3D games. </b>

<p align="center">
  <img src="images/output_grid.gif" alt="Full Width GIF" width="100%" />
</p>


## Latest updates
- **03/21/2025 -- We have released the codebase for ROCKET-2!**

## Docker

```sh
docker build -t rocket2 .
docker run -it -p 7860:7860 --platform=linux/amd64 --gpus all rocket2:latest
```

## Gradio

![](images/demo.png)

## Citing ROCKET-2
If you use ROCKET-2 in your research, please use the following BibTeX entry. 

```
@article{cai2025rocket,
  title={ROCKET-2: Steering Visuomotor Policy via Cross-View Goal Alignment},
  author={Cai, Shaofei and Mu, Zhancun and Liu, Anji and Liang, Yitao},
  journal={arXiv preprint arXiv:2503.02505},
  year={2025}
}
```
