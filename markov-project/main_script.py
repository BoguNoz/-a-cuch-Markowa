#%%

# Stworzyć i zaprezentować proces łańcuchu Markowa.
# Zrobić to dla dwu wymiarowego wektora.
# Korzystając z tej implementacji przejść do procesu Browana.

#%%

from markov_1d_module import sample_markov_result_showcase_1d
from markov_2d_module import sample_markov_result_showcase_2d
from brown_module import  sample_brownian_result_showcase_1d, sample_brownian_result_showcase_2d

steps = 100

sample_markov_result_showcase_1d(steps)
sample_markov_result_showcase_2d(steps)

sample_brownian_result_showcase_1d(steps)
sample_brownian_result_showcase_2d(steps)