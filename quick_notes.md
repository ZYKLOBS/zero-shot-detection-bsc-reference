- Uses Super-Resolution based lossless compressor (SReC) Cao, S., Wu, C.Y., Krähenbühl, P.: Lossless Image Compression through Super-
    Resolution. arXiv preprint arXiv:2004.02872v1 (2020)

- In Superresolution case (Equation 6), why do they use K = 10? Maybe change this hyperparameter

- Options to change model
    1. Change SReC to another lossless image compression algorithm
    2. Maybe use slower but more exact algorithm? Research! (Autoregressive, VAE, Flow-based, diffusion models according to chatgpt)
    3.

- verify if assumption holds -> must if paper correct Our fundamental
    assumption is that the trained CNNs provide a good model of real images, and
    synthetic images tend not to follow the same model

- Can we use other test statistics?
    Therefore, to get rid of this bias, we consider the coding cost
    gap, defined as the difference D(l) = NLL(l)  H(l), as decision statistic. (page 9)
    
    Therefore, as
        decision statistics we will consider both D(0) (the level-0 coding cost gap) and
        ∆01 = D(0) − D(1) (its slope). I  Therefore, besides the above statistics we also consider their abso-
lute values D(0) and ∆(01) .


- Our work was developed to detect whether an image has been fully generated
and not to detect local manipulations. However, it could be easily extended
to accomplish this task since we already compute a map of local pixel-wise
statistics. Furthermore, our approach relies on a model of the real class learned
by the encoder. If real images do not satisfy this model, the approach may not
perform correctly. For example, if images are highly compressed or resized (as is
the case on the web), statistical analysis may not be reliable

Vielleicht das interessanter? 
- this suggests
there may be better ways to exploit the basic NLL(l) and H(l), possibly jointly
at all levels, to synthesize a better and more stable decision statistics. (Page 13)


- 30 datasets  

-Future work will focus on making the method robust to
the most common forms of image impairment, so as to make it suitable for in
the wild application.
