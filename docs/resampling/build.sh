#!/bin/csh

setenv TEXINPUTS /home/krosenfe/School/tikz-dsp//:
latex --shell-escape summary
bibtex summary
latex --shell-escape summary
pdflatex --shell-escape summary
acroread summary.pdf
