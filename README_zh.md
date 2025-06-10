# robosuite

![环境画廊](docs/images/gallery.png)

[**[主页]**](https://robosuite.ai/) &ensp; [**[白皮书]**](https://arxiv.org/abs/2009.12293) &ensp; [**[文档]**](https://robosuite.ai/docs/overview.html) &ensp; [**[ARISE 计划]**](https://github.com/ARISE-Initiative)

-------

## 最新动态

- [2024/10/28] **v1.5**：新增多样化机器人形态（包括人形机器人）、自定义机器人组合、复合控制器（包括全身控制器）、更多远程操作设备、照片级真实渲染。[[发布说明]](https://github.com/ARISE-Initiative/robosuite/releases/tag/v1.5.0) [[文档]](http://robosuite.ai/docs/overview.html)

- [2022/11/15] **v1.4**：后端迁移至 DeepMind 官方 [MuJoCo Python 绑定](https://github.com/deepmind/mujoco)，支持机器人材质和修复若干 bug :robot: [[发布说明]](https://github.com/ARISE-Initiative/robosuite/releases/tag/v1.4.0) [[文档]](http://robosuite.ai/docs/v1.4/)

- [2021/10/19] **v1.3**：新增光线追踪和基于物理的渲染工具 :sparkles:，并支持更多视觉模态 🎥 [[视频介绍]](https://www.youtube.com/watch?v=2xesly6JrQ8) [[发布说明]](https://github.com/ARISE-Initiative/robosuite/releases/tag/v1.3) [[文档]](http://robosuite.ai/docs/v1.3/)

- [2021/02/17] **v1.2**：新增可观测传感器模型 :eyes: 和动力学随机化 :game_die: [[发布说明]](https://github.com/ARISE-Initiative/robosuite/releases/tag/v1.2)

- [2020/12/17] **v1.1**：重构基础架构，标准化模型类，便于环境原型开发 :wrench: [[发布说明]](https://github.com/ARISE-Initiative/robosuite/releases/tag/v1.1)

-------

**robosuite** 是一个基于 [MuJoCo](http://mujoco.org/) 物理引擎的机器人学习仿真框架，并提供了一套可复现的基准环境。当前版本（v1.5）支持多样化机器人形态（包括人形机器人）、自定义机器人组合、复合控制器（包括全身控制器）、更多远程操作设备、照片级真实渲染。本项目隶属于 [ARISE 计划](https://github.com/ARISE-Initiative)，旨在降低 AI 与机器人交叉领域前沿研究的门槛。

数据驱动的算法（如强化学习和模仿学习）为机器人领域提供了强大且通用的工具。随着深度学习的进步，这些学习范式已在多种机器人控制问题中取得了令人兴奋的成果。然而，可复现性挑战和机器人硬件的有限可获得性（尤其在疫情期间）阻碍了研究进展。**robosuite** 的核心目标是为研究者提供：

* 一套标准化的基准任务，用于严格评估和算法开发；
* 灵活的模块化设计，便于构建新的机器人仿真环境；
* 高质量的机器人控制器实现和开箱即用的学习算法，降低入门门槛。

该框架最初由 [斯坦福视觉与学习实验室](http://svl.stanford.edu)（SVL）于 2017 年末开发，现已在 SVL、[德州大学机器人感知与学习实验室](http://rpl.cs.utexas.edu)（RPL）以及 NVIDIA [通用体智能体研究组](https://research.nvidia.com/labs/gear/)（GEAR）等多个研究团队中广泛应用和维护。我们欢迎社区贡献，详情请参阅[贡献指南](CONTRIBUTING.md)。

**robosuite** 提供了模块化的 API，可用于构建新环境、机器人形态和控制器，支持过程生成。主要特性包括：

* **标准化任务**：多样且复杂度各异的标准化操作任务，并提供 RL 基准结果，便于可复现研究；
* **过程生成**：模块化 API，支持以编程方式组合机器人模型、场地和参数化 3D 物体，快速创建新环境和任务。更多机器人模型请见 [robosuite_models](https://github.com/ARISE-Initiative/robosuite_models)；
* **机器人控制器**：多种控制器类型，包括关节空间速度控制、逆运动学控制、操作空间控制和全身控制；
* **远程操作设备**：支持键盘、spacemouse 及 MuJoCo 可视化拖拽等多种远程操作方式；
* **多模态传感器**：支持多种传感信号，包括低层物理状态、RGB 相机、深度图和本体感知；
* **人类演示**：提供采集、回放和利用人类演示数据的工具。相关项目请见 [robomimic](https://arise-initiative.github.io/robomimic-web/)；
* **照片级真实渲染**：集成先进图形工具，支持实时照片级真实渲染，包括对 NVIDIA Isaac Sim 渲染的支持。

## 引用

如果您在论文或项目中使用了 **robosuite**，请引用如下文献：

```bibtex
@inproceedings{robosuite2020,
  title={robosuite: A Modular Simulation Framework and Benchmark for Robot Learning},
  author={Yuke Zhu and Josiah Wong and Ajay Mandlekar and Roberto Mart\'{i}n-Mart\'{i}n and Abhishek Joshi and Soroush Nasiriany and Yifeng Zhu and Kevin Lin},
  booktitle={arXiv preprint arXiv:2009.12293},
  year={2020}
}
``` 