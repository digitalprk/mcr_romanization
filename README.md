Automated McCune-Reischauer Romanization and Word Division for Korean Text
=====

* Automatically segments a Hangul sentence into its constituent parts of speech following the McCune-Reischauer rules for word segmentation
* Romanize each word according to McCune-Reischauer transliteration rules 
* A sequence-to-sequence neural model is used for romanization. Conditional random fields are used for word segmentation.
* All training data is available in the data folder

Information
----

* For information on McCune-Reischauer Romanization : https://www.loc.gov/catdir/cpso/romanization/korean.pdf
* For information on this project : http://digitalnk.com/blog/2017/12/23/neural-networks-and-the-bane-of-romanization/
* Online beta: http://www.digitalnk.com/romanizer/

Usage
----

```
from romanizer import Romanizer

r = Romanizer()

r.Segment('생물학적 죽음에서 인간적 죽음으로')
Out[57]: '생물학적 죽음 에서 인간적 죽음 으로'

r.Romanize('작가의 정체성과 개작, 그리고 평가 : 황순원 "움직이는 성"의 개작을 중심으로')
Out[32]: "chakka ŭi chŏngch'esŏng kwa kaejak kŭrigo p'yŏngka : hwang sunwŏn umjiginŭn sŏng ŭi kaejak ŭl chungsim ŭro"
```
