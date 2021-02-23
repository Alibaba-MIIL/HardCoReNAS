# HardCoRe-NAS: Hard Constrained diffeRentiableNeural Architecture Search
Code accompanying the paper:
> [HardCoRe-NAS: Hard Constrained diffeRentiableNeural Architecture Search](https://arxiv.org/abs/???)\
> Niv Nayman, Yonathan Aflalo, Asaf Noy, Lihi Zelnik-Manor.\
> _arXiv:???_.

Realistic use of neural networks often requires adhering to multiple constraints on latency, energy and memory among others.
A popular approach to find fitting networks is through constrained Neural Architecture Search (NAS), however, previous methods enforce the constraint only softly.
Therefore, the resulting networks do not exactly adhere to the resource constraint and their accuracy is harmed.
In this work we resolve this by introducing Hard Constrained diffeRentiable NAS (HardCoRe-NAS), that is based on an accurate formulation of the expected resource requirement and a scalable search method that satisfies the hard constraint throughout the search.
Our experiments show that HardCoRe-NAS generates state-of-the-art architectures, surpassing other NAS methods, while strictly satisfying the hard resource constraints without any tuning required.

<object data="https://github.com/Alibaba-MIIL/HardCoReNAS/blob/main/images/hardcorenas_system.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="https://github.com/Alibaba-MIIL/HardCoReNAS/blob/main/images/hardcorenas_system.pdf">
	<p align="center">
	    <img src="images/hardcorenas_system.png" alt="hardcorenas_system" width="80%">
	</p>
    </embed>
</object>

### WIP
The code is being prepared for a public release - watch this project to be notified.
