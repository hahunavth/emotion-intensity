_punctuation = "!'(),.:;?"
_letters = "t aw1 aa6 nc e3 wa1 iz4 oz6 uw6 e1 oz4 oz3 mc u1 ee1 aa5 p ooo3 kh aa1 uw5 ie1 o4 w o2 tr chz iz6 uw3 ow5 e4 m e5 a1 uw2 uw1 a3 uz1 wa4 ee5 b nh iz3 h ng oz1 o3 oo4 i4 uo1 uo5 nhz ee6 i6 i3 th iz1 ph aw2 a6 uo2 uo6 a5 aa2 v a2 wa3 o6 g x ch oo5 ie2 oo1 kc ie3 ie4 ow4 aw4 aw3 k ooo2 ooo1 uz3 a4 ow3 oo3 uw4 aa3 o5 iz2 aw6 i5 uz6 n wa5 i1 ooo5 uz2 ee4 ow2 ee2 ow6 wa2 e2 d iz5 u5 pc r u3 aa4 ngz u4 uz5 u2 uz4 l oo2 aw5 oz2 tc u6 ooo6 ee3 oo6 ie5 dd uo3 uo4 oz5 ow1 ie6 i2 wa6 ooo4 e6 o1"
_letters = _letters.split()
_silences = ["spn", "sil", "sp"]
_space = ["spc", "dot"]

symbols = (
    _space
    + list(_punctuation)
    + _silences
    + _letters
)

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

def phoneme_to_ids(text):
    phoneme = text.strip("{").strip("}")
    # phonemes = " spc ".join(phoneme.split()).split()
    phonemes = phoneme.split()
    phoneme_ids = [_symbol_to_id[s] for s in phonemes]
    
    return phoneme_ids

def ids_to_phoneme(ids):
    phoneme_to_ids = []
    
    for _id in ids:
        phoneme_to_ids.append(_id_to_symbol[_id])
    
    return " ".join(phoneme_to_ids)