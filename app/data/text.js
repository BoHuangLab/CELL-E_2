const abstract =
  `We present methodname, a novel bidirectional non-autoregressive transformer that can generate realistic images and sequences of protein localization in the cell. Protein localization is a challenging problem that requires integrating sequence and image information, which most existing methods ignore. methodname extends the work of CELL-E by capturing the spatial complexity of protein localization and produce probability estimates of localization atop a nucleus image, but can also generate sequences from images, enabling de novo protein design. We train and finetune methodname on two large-scale datasets of human proteins. We also demonstrate how to use methodname to create hundreds of novel nuclear localization signals (NLS) for Green Fluorescent Protein (GFP).
`
    .split('.')
    .map(item => item.replaceAll('methodname', 'CELL-E 2'));

export { abstract };
