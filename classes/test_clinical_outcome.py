import numpy as np
from clinical_outcome import Clinical_outcome

outcome = Clinical_outcome()

#mimic = np.array([0])
#ich = np.array([0.16])
#nlvo = np.array([0.52])
#lvo = np.array([0.32])
#onset_to_needle = np.array([100])
#onset_to_puncture =np.array([150])
#nlvo_eligible_for_treatment = np.array([0.194])
#lvo_eligible_for_treatment = np.array([0.313])
#prop_thrombolysed_lvo_receiving_thrombectomy = np.array([1])

mimic = np.array([0]*21)
ich = np.array([0.16]*21)
nlvo = np.array([0.52]*21)
lvo = np.array([0.32]*21)
onset_to_needle = np.linspace(0,600,21)
onset_to_puncture = onset_to_needle + 360 # np.linspace(0,600,21)
nlvo_eligible_for_treatment = np.array([1]*21)
lvo_eligible_for_treatment = np.array([1]*21)
prop_thrombolysed_lvo_receiving_thrombectomy = np.array([1]*21)

outcomes=outcome.calculate_outcome_for_all(
    mimic,
    ich,
    nlvo,
    lvo,
    onset_to_needle,
    onset_to_puncture,
    nlvo_eligible_for_treatment,
    lvo_eligible_for_treatment,
    prop_thrombolysed_lvo_receiving_thrombectomy)

print (outcomes)

np.savetxt("foo.csv", outcomes, delimiter=",")

import matplotlib.pyplot as plt
plt.plot(onset_to_puncture, outcomes)
plt.show()
