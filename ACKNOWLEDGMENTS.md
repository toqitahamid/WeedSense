# Acknowledgments

## Base Framework

This project is built upon the excellent [semantic-segmentation](https://github.com/sithu31296/semantic-segmentation) library by [@sithu31296](https://github.com/sithu31296). We are grateful for their comprehensive implementation of state-of-the-art semantic segmentation models in PyTorch, which provided a solid foundation for our multi-task learning architecture.

**Original Repository**: https://github.com/sithu31296/semantic-segmentation  
**License**: MIT License

## Key Adaptations

We adapted and extended the base library with:

1. **Multi-Task Learning Framework**: Added support for simultaneous segmentation, height estimation, and growth stage classification
2. **UIB Encoder Architecture**: Integrated Universal Inverted Bottleneck blocks with Squeeze-and-Excitation modules
3. **Multi-Task Bifurcated Decoder**: Designed a novel decoder with transformer-based feature fusion for temporal growth analysis
4. **Agricultural Dataset Support**: Implemented custom dataset loader for weed analysis with multi-modal annotations
5. **Auxiliary Supervision**: Added multiple auxiliary heads for improved gradient flow during training

## Research Support

This research was conducted at:
- **Southern Illinois University Carbondale**
- **SIU Horticulture Research Center**

## Open Source Tools

We thank the developers of:
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [timm](https://github.com/huggingface/pytorch-image-models) - PyTorch image models
- [SAM2](https://github.com/facebookresearch/segment-anything-2) - Annotation assistance
- [BiSeNetV2](https://arxiv.org/abs/2004.02147) - Real-time semantic segmentation

## Funding & Support

This work was supported by Southern Illinois University Carbondale and the SIU Horticulture Research Center.

## Citation

If you use this work, please cite both our paper and the base library:

### WeedSense (Our Work)
```bibtex
@article{sarker2025weedsense,
  title={WeedSense: Multi-Task Learning for Weed Segmentation, Height Estimation, and Growth Stage Classification},
  author={Sarker, Toqi Tahamid and Ahmed, Khaled R and Islam, Taminul and Rankrape, Cristiana Bernardi and Gage, Karla},
  journal={arXiv preprint arXiv:2508.14486},
  year={2025}
}
```

### Base Library
```bibtex
@misc{sithu2022semseg,
  title={Semantic Segmentation in PyTorch},
  author={Sithu Aung},
  year={2022},
  howpublished={\url{https://github.com/sithu31296/semantic-segmentation}},
}
```

## Contact

For questions about this implementation:
- **Email**: toqitahamid.sarker@siu.edu
- **Project Page**: https://weedsense.github.io
- **GitHub**: https://github.com/toqitahamid/weedsense
