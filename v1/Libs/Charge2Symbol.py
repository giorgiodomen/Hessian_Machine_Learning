from numpy import pi
ELEMENTS = ['X',  # Ghost
    'H' , 'He', 'Li', 'Be', 'B' , 'C' , 'N' , 'O' , 'F' , 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P' , 'S' , 'Cl', 'Ar', 'K' , 'Ca',
    'Sc', 'Ti', 'V' , 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y' , 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I' , 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W' , 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U' , 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og',
]

NUC = dict(((x,i) for i,x in enumerate(ELEMENTS)))
NUC.update((x.upper(),i) for i,x in enumerate(ELEMENTS))
NUC['GHOST'] = 0
ELEMENTS_PROTON = NUC

def _rm_digit(symb):
    if symb.isalpha():
        return symb
    else:
        return ''.join([i for i in symb if i.isalpha()])
def charge(symb_or_chg):
    if isinstance(symb_or_chg, (str)):
        a = symb_or_chg.upper()
        if ('GHOST' in a or ('X' in a and 'XE' not in a)):
            return 0
        else:
            return ELEMENTS_PROTON[str(_rm_digit(a))]
    else:
        return symb_or_chg

ang2bohr=1.8897261246
bohr2ang=.5291772109
centimeter2bohr=1.8897261246e+8
plankAU=2*pi
lightspeedAU=137.036
dalton_to_au=  1.660e-27 / 9.109e-31

def to_cm(k,Mu):
    return (k/Mu)**0.5/plankAU/lightspeedAU*centimeter2bohr
