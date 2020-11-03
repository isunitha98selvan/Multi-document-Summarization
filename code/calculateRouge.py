# progeram to calculate Rouge scores
from PyRouge.pyrouge import Rouge

r = Rouge()

system_generated_summary="In Italy, Dini told reporters in the city of Trento that the government would decide Ocalan's request for asylum in a clear and responsible manner''but Ocalan has been absent from the battlefield for at least a month, and others contend that the PKK is capable of operating without him. Facing his first real foreign policy test, Prime Minister Massimo D'Alema must decide what to do with a prominent Kurdish rebel leader who was arrested at the Rome airport on Thursday"
manual_summmary = "An international incident resulted after the arrest in Rome of Abdullah Ocalan, the leader of the Kurdistan Workers Party (PKK). An armed struggle has ensued since the PKK was formed in 1978, and nearly 37,000 died. Turkey wants Ocalan extradited but Italy is reluctant since Turkey still has the death penalty. Kurds from all over Europe have come to Rome, or at least tried to, to protest Ocalan's detention and to urge asylum for him. Turkey has said countries bordering eastern Turkey have harbored Kurdish rebels and Greece has voiced support for the Kurds. Prisoners in Turkey held an Italian inmate hostage in hope of forcing Italy to extradite Ocalan."

[precision, recall, f_score] = r.rouge_l([system_generated_summary], [manual_summmary])

print("Precision is :"+str(precision)+"\nRecall is :"+str(recall)+"\nF Score is :"+str(f_score))