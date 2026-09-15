# name, SMILES, group
SET_A = [
 # --- annularity / ring-topology series -------------------------------
 ("ethene",            "C=C",                                   "reference"),
 ("butadiene",         "C=CC=C",                                "reference"),
 ("hexatriene",        "C=CC=CC=C",                             "reference"),
 ("benzene",           "c1ccccc1",                              "acene"),
 ("naphthalene",       "c1ccc2ccccc2c1",                        "acene"),
 ("anthracene",        "c1ccc2cc3ccccc3cc2c1",                  "acene"),
 ("tetracene",         "c1ccc2cc3cc4ccccc4cc3cc2c1",            "acene"),
 ("pentacene",         "c1ccc2cc3cc4cc5ccccc5cc4cc3cc2c1",      "acene"),
 ("phenanthrene",      "c1ccc2c(c1)ccc1ccccc21",                "PAH"),
 ("pyrene",            "c1cc2ccc3cccc4ccc(c1)c2c34",            "PAH"),
 ("triphenylene",      "c1ccc2c(c1)c1ccccc1c1ccccc21",          "PAH"),
 ("perylene",          "c1cc2cccc3c2c2c(c1)cccc2c1cccc31",      "PAH"),
 ("coronene",          "c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61","PAH"),
 ("benzo[a]pyrene",    "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",      "PAH"),
 ("azulene",           "C1=CC2=CC=CC=CC2=C1",                   "PAH"),
 ("biphenyl",          "c1ccc(-c2ccccc2)cc1",                   "biaryl"),
 # --- heteroaromatics --------------------------------------------------
 ("pyridine",          "c1ccncc1",                              "heteroarene"),
 ("pyrimidine",        "c1cncnc1",                              "heteroarene"),
 ("s-triazine",        "c1ncncn1",                              "heteroarene"),
 ("pyrrole",           "c1cc[nH]c1",                            "heteroarene"),
 ("furan",             "c1ccoc1",                               "heteroarene"),
 ("thiophene",         "c1ccsc1",                               "heteroarene"),
 ("imidazole",         "c1c[nH]cn1",                            "heteroarene"),
 ("indole",            "c1ccc2[nH]ccc2c1",                      "heteroarene"),
 ("quinoline",         "c1ccc2ncccc2c1",                        "heteroarene"),
 ("porphine",          "C1=CC2=CC3=NC(=CC4=CC=C(N4)C=C4C=CC(=N4)C=C1N2)C=C3", "macrocycle"),
 # --- steric shielding series (constant ring count) --------------------
 ("toluene",           "Cc1ccccc1",                             "shielding"),
 ("p-xylene",          "Cc1ccc(C)cc1",                          "shielding"),
 ("mesitylene",        "Cc1cc(C)cc(C)c1",                       "shielding"),
 ("hexamethylbenzene", "Cc1c(C)c(C)c(C)c(C)c1C",                "shielding"),
 ("hexaethylbenzene",  "CCc1c(CC)c(CC)c(CC)c(CC)c1CC",          "shielding"),
 ("1,3,5-tri-tBu-benzene","CC(C)(C)c1cc(C(C)(C)C)cc(C(C)(C)C)c1","shielding"),
 # --- substituent electronic series ------------------------------------
 ("fluorobenzene",     "Fc1ccccc1",                             "substituted"),
 ("hexafluorobenzene", "Fc1c(F)c(F)c(F)c(F)c1F",                "substituted"),
 ("phenol",            "Oc1ccccc1",                             "substituted"),
 ("aniline",           "Nc1ccccc1",                             "substituted"),
 ("nitrobenzene",      "O=[N+]([O-])c1ccccc1",                  "substituted"),
 ("benzonitrile",      "N#Cc1ccccc1",                           "substituted"),
 # --- curved / non-planar ----------------------------------------------
 ("corannulene",       "c1cc2ccc3ccc4ccc5ccc1c1c2c3c4c51",      "curved"),
]

# monomers used for the stacking-energy benchmark (kept small for cost)
SET_B = [
 "benzene","toluene","p-xylene","mesitylene","hexamethylbenzene",
 "fluorobenzene","hexafluorobenzene","phenol","aniline","nitrobenzene",
 "benzonitrile","pyridine","pyrimidine","s-triazine","pyrrole","furan",
 "thiophene","imidazole","indole","quinoline","naphthalene","azulene",
 "phenanthrene","anthracene","biphenyl","butadiene","hexatriene","ethene",
]
