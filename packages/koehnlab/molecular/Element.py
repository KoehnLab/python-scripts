from typing import Optional

from enum import Enum


class Element(Enum):
    """Enumeration listing all known elements"""

    H = 1
    Hydrogen = 1
    He = 2
    Helium = 2
    Li = 3
    Lithium = 3
    Be = 4
    Beryllium = 4
    B = 5
    Boron = 5
    C = 6
    Carbon = 6
    N = 7
    Nitrogen = 7
    O = 8
    Oxygen = 8
    F = 9
    Fluorine = 9
    Ne = 10
    Neon = 10
    Na = 11
    Sodium = 11
    Mg = 12
    Magnesium = 12
    Al = 13
    Aluminium = 13
    Si = 14
    Silicon = 14
    P = 15
    Phosphorus = 15
    S = 16
    Sulfur = 16
    Cl = 17
    Chlorine = 17
    Ar = 18
    Argon = 18
    K = 19
    Potassium = 19
    Ca = 20
    Calcium = 20
    Sc = 21
    Scandium = 21
    Ti = 22
    Titanium = 22
    V = 23
    Vanadium = 23
    Cr = 24
    Chromium = 24
    Mn = 25
    Manganese = 25
    Fe = 26
    Iron = 26
    Co = 27
    Cobalt = 27
    Ni = 28
    Nickel = 28
    Cu = 29
    Copper = 29
    Zn = 30
    Zinc = 30
    Ga = 31
    Gallium = 31
    Ge = 32
    Germanium = 32
    As = 33
    Arsenic = 33
    Se = 34
    Selenium = 34
    Br = 35
    Bromine = 35
    Kr = 36
    Krypton = 36
    Rb = 37
    Rubidium = 37
    Sr = 38
    Strontium = 38
    Y = 39
    Yttrium = 39
    Zr = 40
    Zirconium = 40
    Nb = 41
    Niobium = 41
    Mo = 42
    Molybdenum = 42
    Tc = 43
    Technetium = 43
    Ru = 44
    Ruthenium = 44
    Rh = 45
    Rhodium = 45
    Pd = 46
    Palladium = 46
    Ag = 47
    Silver = 47
    Cd = 48
    Cadmium = 48
    In = 49
    Indium = 49
    Sn = 50
    Tin = 50
    Sb = 51
    Antimony = 51
    Te = 52
    Tellurium = 52
    I = 53
    Iodine = 53
    Xe = 54
    Xenon = 54
    Cs = 55
    Caesium = 55
    Ba = 56
    Barium = 56
    La = 57
    Lanthanum = 57
    Ce = 58
    Cerium = 58
    Pr = 59
    Praseodymium = 59
    Nd = 60
    Neodymium = 60
    Pm = 61
    Promethium = 61
    Sm = 62
    Samarium = 62
    Eu = 63
    Europium = 63
    Gd = 64
    Gadolinium = 64
    Tb = 65
    Terbium = 65
    Dy = 66
    Dysprosium = 66
    Ho = 67
    Holmium = 67
    Er = 68
    Erbium = 68
    Tm = 69
    Thulium = 69
    Yb = 70
    Ytterbium = 70
    Lu = 71
    Lutetium = 71
    Hf = 72
    Hafnium = 72
    Ta = 73
    Tantalum = 73
    W = 74
    Tungsten = 74
    Re = 75
    Rhenium = 75
    Os = 76
    Osmium = 76
    Ir = 77
    Iridium = 77
    Pt = 78
    Platinum = 78
    Au = 79
    Gold = 79
    Hg = 80
    Mercury = 80
    Tl = 81
    Thallium = 81
    Pb = 82
    Lead = 82
    Bi = 83
    Bismuth = 83
    Po = 84
    Polonium = 84
    At = 85
    Astatine = 85
    Rn = 86
    Radon = 86
    Fr = 87
    Francium = 87
    Ra = 88
    Radium = 88
    Ac = 89
    Actinium = 89
    Th = 90
    Thorium = 90
    Pa = 91
    Protactinium = 91
    U = 92
    Uranium = 92
    Np = 93
    Neptunium = 93
    Pu = 94
    Plutonium = 94
    Am = 95
    Americium = 95
    Cm = 96
    Curium = 96
    Bk = 97
    Berkelium = 97
    Cf = 98
    Californium = 98
    Es = 99
    Einsteinium = 99
    Fm = 100
    Fermium = 100
    Md = 101
    Mendelevium = 101
    No = 102
    Nobelium = 102
    Lr = 103
    Lawrencium = 103
    Rf = 104
    Rutherfordium = 104
    Db = 105
    Dubnium = 105
    Sg = 106
    Seaborgium = 106
    Bh = 107
    Bohrium = 107
    Hs = 108
    Hassium = 108
    Mt = 109
    Meitnerium = 109
    Ds = 110
    Darmstadtium = 110
    Rg = 111
    Roentgenium = 111
    Cn = 112
    Copernicium = 112
    Nh = 113
    Nihonium = 113
    Fl = 114
    Flerovium = 114
    Mc = 115
    Moscovium = 115
    Lv = 116
    Livermorium = 116
    Ts = 117
    Tennessine = 117
    Og = 118
    Oganesson = 118

    def symbol(self) -> str:
        """Gets the element symbol"""
        return self.name

    def fullName(self) -> str:
        """Gets the full name of this Element (English)"""
        return ElementNames[self]

    def atomicNumber(self) -> int:
        """Gets this element's atomic number"""
        return self.value

    def mass(self) -> float:
        """Gets the standard mass of this Element (in Dalton)"""
        return ElementMasses[self]

    def group(self) -> Optional[int]:
        """Gets the element's group. If the element belongs to the f-block, this will
        return None as these elements don't really belong to any group"""
        return ElementGroups[self]

    def period(self) -> int:
        """Gets the period to which this element belongs"""
        return ElementPeriods[self]

    def block(self) -> str:
        """Gets the block to which this element belongs to. This is one of "s", "p", "d" or "f"."""
        return ElementBlocks[self]


ElementNames = {
    Element.H: "Hydrogen",
    Element.He: "Helium",
    Element.Li: "Lithium",
    Element.Be: "Beryllium",
    Element.B: "Boron",
    Element.C: "Carbon",
    Element.N: "Nitrogen",
    Element.O: "Oxygen",
    Element.F: "Fluorine",
    Element.Ne: "Neon",
    Element.Na: "Sodium",
    Element.Mg: "Magnesium",
    Element.Al: "Aluminium",
    Element.Si: "Silicon",
    Element.P: "Phosphorus",
    Element.S: "Sulfur",
    Element.Cl: "Chlorine",
    Element.Ar: "Argon",
    Element.K: "Potassium",
    Element.Ca: "Calcium",
    Element.Sc: "Scandium",
    Element.Ti: "Titanium",
    Element.V: "Vanadium",
    Element.Cr: "Chromium",
    Element.Mn: "Manganese",
    Element.Fe: "Iron",
    Element.Co: "Cobalt",
    Element.Ni: "Nickel",
    Element.Cu: "Copper",
    Element.Zn: "Zinc",
    Element.Ga: "Gallium",
    Element.Ge: "Germanium",
    Element.As: "Arsenic",
    Element.Se: "Selenium",
    Element.Br: "Bromine",
    Element.Kr: "Krypton",
    Element.Rb: "Rubidium",
    Element.Sr: "Strontium",
    Element.Y: "Yttrium",
    Element.Zr: "Zirconium",
    Element.Nb: "Niobium",
    Element.Mo: "Molybdenum",
    Element.Tc: "Technetium",
    Element.Ru: "Ruthenium",
    Element.Rh: "Rhodium",
    Element.Pd: "Palladium",
    Element.Ag: "Silver",
    Element.Cd: "Cadmium",
    Element.In: "Indium",
    Element.Sn: "Tin",
    Element.Sb: "Antimony",
    Element.Te: "Tellurium",
    Element.I: "Iodine",
    Element.Xe: "Xenon",
    Element.Cs: "Caesium",
    Element.Ba: "Barium",
    Element.La: "Lanthanum",
    Element.Ce: "Cerium",
    Element.Pr: "Praseodymium",
    Element.Nd: "Neodymium",
    Element.Pm: "Promethium",
    Element.Sm: "Samarium",
    Element.Eu: "Europium",
    Element.Gd: "Gadolinium",
    Element.Tb: "Terbium",
    Element.Dy: "Dysprosium",
    Element.Ho: "Holmium",
    Element.Er: "Erbium",
    Element.Tm: "Thulium",
    Element.Yb: "Ytterbium",
    Element.Lu: "Lutetium",
    Element.Hf: "Hafnium",
    Element.Ta: "Tantalum",
    Element.W: "Tungsten",
    Element.Re: "Rhenium",
    Element.Os: "Osmium",
    Element.Ir: "Iridium",
    Element.Pt: "Platinum",
    Element.Au: "Gold",
    Element.Hg: "Mercury",
    Element.Tl: "Thallium",
    Element.Pb: "Lead",
    Element.Bi: "Bismuth",
    Element.Po: "Polonium",
    Element.At: "Astatine",
    Element.Rn: "Radon",
    Element.Fr: "Francium",
    Element.Ra: "Radium",
    Element.Ac: "Actinium",
    Element.Th: "Thorium",
    Element.Pa: "Protactinium",
    Element.U: "Uranium",
    Element.Np: "Neptunium",
    Element.Pu: "Plutonium",
    Element.Am: "Americium",
    Element.Cm: "Curium",
    Element.Bk: "Berkelium",
    Element.Cf: "Californium",
    Element.Es: "Einsteinium",
    Element.Fm: "Fermium",
    Element.Md: "Mendelevium",
    Element.No: "Nobelium",
    Element.Lr: "Lawrencium",
    Element.Rf: "Rutherfordium",
    Element.Db: "Dubnium",
    Element.Sg: "Seaborgium",
    Element.Bh: "Bohrium",
    Element.Hs: "Hassium",
    Element.Mt: "Meitnerium",
    Element.Ds: "Darmstadtium",
    Element.Rg: "Roentgenium",
    Element.Cn: "Copernicium",
    Element.Nh: "Nihonium",
    Element.Fl: "Flerovium",
    Element.Mc: "Moscovium",
    Element.Lv: "Livermorium",
    Element.Ts: "Tennessine",
    Element.Og: "Oganesson",
}


# Values copied from Wikipedia (https://en.wikipedia.org/wiki/List_of_chemical_elements) so probably not most accurate
ElementMasses = {
    Element.H: 1.008,
    Element.He: 4.0026,
    Element.Li: 6.94,
    Element.Be: 9.0122,
    Element.B: 10.81,
    Element.C: 12.011,
    Element.N: 14.007,
    Element.O: 15.999,
    Element.F: 18.998,
    Element.Ne: 20.18,
    Element.Na: 22.99,
    Element.Mg: 24.305,
    Element.Al: 26.982,
    Element.Si: 28.085,
    Element.P: 30.974,
    Element.S: 32.06,
    Element.Cl: 35.45,
    Element.Ar: 39.95,
    Element.K: 39.098,
    Element.Ca: 40.078,
    Element.Sc: 44.956,
    Element.Ti: 47.867,
    Element.V: 50.942,
    Element.Cr: 51.996,
    Element.Mn: 54.938,
    Element.Fe: 55.845,
    Element.Co: 58.933,
    Element.Ni: 58.693,
    Element.Cu: 63.546,
    Element.Zn: 65.38,
    Element.Ga: 69.723,
    Element.Ge: 72.63,
    Element.As: 74.922,
    Element.Se: 78.971,
    Element.Br: 79.904,
    Element.Kr: 83.798,
    Element.Rb: 85.468,
    Element.Sr: 87.62,
    Element.Y: 88.906,
    Element.Zr: 91.224,
    Element.Nb: 92.906,
    Element.Mo: 95.95,
    Element.Tc: 97,
    Element.Ru: 101.07,
    Element.Rh: 102.91,
    Element.Pd: 106.42,
    Element.Ag: 107.87,
    Element.Cd: 112.41,
    Element.In: 114.82,
    Element.Sn: 118.71,
    Element.Sb: 121.76,
    Element.Te: 127.6,
    Element.I: 126.9,
    Element.Xe: 131.29,
    Element.Cs: 132.91,
    Element.Ba: 137.33,
    Element.La: 138.91,
    Element.Ce: 140.12,
    Element.Pr: 140.91,
    Element.Nd: 144.24,
    Element.Pm: 145,
    Element.Sm: 150.36,
    Element.Eu: 151.96,
    Element.Gd: 157.25,
    Element.Tb: 158.93,
    Element.Dy: 162.5,
    Element.Ho: 164.93,
    Element.Er: 167.26,
    Element.Tm: 168.93,
    Element.Yb: 173.05,
    Element.Lu: 174.97,
    Element.Hf: 178.49,
    Element.Ta: 180.95,
    Element.W: 183.84,
    Element.Re: 186.21,
    Element.Os: 190.23,
    Element.Ir: 192.22,
    Element.Pt: 195.08,
    Element.Au: 196.97,
    Element.Hg: 200.59,
    Element.Tl: 204.38,
    Element.Pb: 207.2,
    Element.Bi: 208.98,
    Element.Po: 209,
    Element.At: 210,
    Element.Rn: 222,
    Element.Fr: 223,
    Element.Ra: 226,
    Element.Ac: 227,
    Element.Th: 232.04,
    Element.Pa: 231.04,
    Element.U: 238.03,
    Element.Np: 237,
    Element.Pu: 244,
    Element.Am: 243,
    Element.Cm: 247,
    Element.Bk: 247,
    Element.Cf: 251,
    Element.Es: 252,
    Element.Fm: 257,
    Element.Md: 258,
    Element.No: 259,
    Element.Lr: 266,
    Element.Rf: 267,
    Element.Db: 268,
    Element.Sg: 269,
    Element.Bh: 270,
    Element.Hs: 269,
    Element.Mt: 278,
    Element.Ds: 281,
    Element.Rg: 282,
    Element.Cn: 285,
    Element.Nh: 286,
    Element.Fl: 289,
    Element.Mc: 290,
    Element.Lv: 293,
    Element.Ts: 294,
    Element.Og: 294,
}


# None means that the element belongs to the f-block groups, which don't seem to really have a group number
# associated with them
ElementGroups = {
    Element.H: 1,
    Element.He: 18,
    Element.Li: 1,
    Element.Be: 2,
    Element.B: 13,
    Element.C: 14,
    Element.N: 15,
    Element.O: 16,
    Element.F: 17,
    Element.Ne: 18,
    Element.Na: 1,
    Element.Mg: 2,
    Element.Al: 13,
    Element.Si: 14,
    Element.P: 15,
    Element.S: 16,
    Element.Cl: 17,
    Element.Ar: 18,
    Element.K: 1,
    Element.Ca: 2,
    Element.Sc: 3,
    Element.Ti: 4,
    Element.V: 5,
    Element.Cr: 6,
    Element.Mn: 7,
    Element.Fe: 8,
    Element.Co: 9,
    Element.Ni: 10,
    Element.Cu: 11,
    Element.Zn: 12,
    Element.Ga: 13,
    Element.Ge: 14,
    Element.As: 15,
    Element.Se: 16,
    Element.Br: 17,
    Element.Kr: 18,
    Element.Rb: 1,
    Element.Sr: 2,
    Element.Y: 3,
    Element.Zr: 4,
    Element.Nb: 5,
    Element.Mo: 6,
    Element.Tc: 7,
    Element.Ru: 8,
    Element.Rh: 9,
    Element.Pd: 10,
    Element.Ag: 11,
    Element.Cd: 12,
    Element.In: 13,
    Element.Sn: 14,
    Element.Sb: 15,
    Element.Te: 16,
    Element.I: 17,
    Element.Xe: 18,
    Element.Cs: 1,
    Element.Ba: 2,
    Element.La: None,
    Element.Ce: None,
    Element.Pr: None,
    Element.Nd: None,
    Element.Pm: None,
    Element.Sm: None,
    Element.Eu: None,
    Element.Gd: None,
    Element.Tb: None,
    Element.Dy: None,
    Element.Ho: None,
    Element.Er: None,
    Element.Tm: None,
    Element.Yb: None,
    Element.Lu: 3,
    Element.Hf: 4,
    Element.Ta: 5,
    Element.W: 6,
    Element.Re: 7,
    Element.Os: 8,
    Element.Ir: 9,
    Element.Pt: 10,
    Element.Au: 11,
    Element.Hg: 12,
    Element.Tl: 13,
    Element.Pb: 14,
    Element.Bi: 15,
    Element.Po: 16,
    Element.At: 17,
    Element.Rn: 18,
    Element.Fr: 1,
    Element.Ra: 2,
    Element.Ac: None,
    Element.Th: None,
    Element.Pa: None,
    Element.U: None,
    Element.Np: None,
    Element.Pu: None,
    Element.Am: None,
    Element.Cm: None,
    Element.Bk: None,
    Element.Cf: None,
    Element.Es: None,
    Element.Fm: None,
    Element.Md: None,
    Element.No: None,
    Element.Lr: 3,
    Element.Rf: 4,
    Element.Db: 5,
    Element.Sg: 6,
    Element.Bh: 7,
    Element.Hs: 8,
    Element.Mt: 9,
    Element.Ds: 10,
    Element.Rg: 11,
    Element.Cn: 12,
    Element.Nh: 13,
    Element.Fl: 14,
    Element.Mc: 15,
    Element.Lv: 16,
    Element.Ts: 17,
    Element.Og: 18,
}


ElementPeriods = {
    Element.H: 1,
    Element.He: 1,
    Element.Li: 2,
    Element.Be: 2,
    Element.B: 2,
    Element.C: 2,
    Element.N: 2,
    Element.O: 2,
    Element.F: 2,
    Element.Ne: 2,
    Element.Na: 3,
    Element.Mg: 3,
    Element.Al: 3,
    Element.Si: 3,
    Element.P: 3,
    Element.S: 3,
    Element.Cl: 3,
    Element.Ar: 3,
    Element.K: 4,
    Element.Ca: 4,
    Element.Sc: 4,
    Element.Ti: 4,
    Element.V: 4,
    Element.Cr: 4,
    Element.Mn: 4,
    Element.Fe: 4,
    Element.Co: 4,
    Element.Ni: 4,
    Element.Cu: 4,
    Element.Zn: 4,
    Element.Ga: 4,
    Element.Ge: 4,
    Element.As: 4,
    Element.Se: 4,
    Element.Br: 4,
    Element.Kr: 4,
    Element.Rb: 5,
    Element.Sr: 5,
    Element.Y: 5,
    Element.Zr: 5,
    Element.Nb: 5,
    Element.Mo: 5,
    Element.Tc: 5,
    Element.Ru: 5,
    Element.Rh: 5,
    Element.Pd: 5,
    Element.Ag: 5,
    Element.Cd: 5,
    Element.In: 5,
    Element.Sn: 5,
    Element.Sb: 5,
    Element.Te: 5,
    Element.I: 5,
    Element.Xe: 5,
    Element.Cs: 6,
    Element.Ba: 6,
    Element.La: 6,
    Element.Ce: 6,
    Element.Pr: 6,
    Element.Nd: 6,
    Element.Pm: 6,
    Element.Sm: 6,
    Element.Eu: 6,
    Element.Gd: 6,
    Element.Tb: 6,
    Element.Dy: 6,
    Element.Ho: 6,
    Element.Er: 6,
    Element.Tm: 6,
    Element.Yb: 6,
    Element.Lu: 6,
    Element.Hf: 6,
    Element.Ta: 6,
    Element.W: 6,
    Element.Re: 6,
    Element.Os: 6,
    Element.Ir: 6,
    Element.Pt: 6,
    Element.Au: 6,
    Element.Hg: 6,
    Element.Tl: 6,
    Element.Pb: 6,
    Element.Bi: 6,
    Element.Po: 6,
    Element.At: 6,
    Element.Rn: 6,
    Element.Fr: 7,
    Element.Ra: 7,
    Element.Ac: 7,
    Element.Th: 7,
    Element.Pa: 7,
    Element.U: 7,
    Element.Np: 7,
    Element.Pu: 7,
    Element.Am: 7,
    Element.Cm: 7,
    Element.Bk: 7,
    Element.Cf: 7,
    Element.Es: 7,
    Element.Fm: 7,
    Element.Md: 7,
    Element.No: 7,
    Element.Lr: 7,
    Element.Rf: 7,
    Element.Db: 7,
    Element.Sg: 7,
    Element.Bh: 7,
    Element.Hs: 7,
    Element.Mt: 7,
    Element.Ds: 7,
    Element.Rg: 7,
    Element.Cn: 7,
    Element.Nh: 7,
    Element.Fl: 7,
    Element.Mc: 7,
    Element.Lv: 7,
    Element.Ts: 7,
    Element.Og: 7,
}


ElementBlocks = {
    Element.H: "s",
    Element.He: "s",
    Element.Li: "s",
    Element.Be: "s",
    Element.B: "p",
    Element.C: "p",
    Element.N: "p",
    Element.O: "p",
    Element.F: "p",
    Element.Ne: "p",
    Element.Na: "s",
    Element.Mg: "s",
    Element.Al: "p",
    Element.Si: "p",
    Element.P: "p",
    Element.S: "p",
    Element.Cl: "p",
    Element.Ar: "p",
    Element.K: "s",
    Element.Ca: "s",
    Element.Sc: "d",
    Element.Ti: "d",
    Element.V: "d",
    Element.Cr: "d",
    Element.Mn: "d",
    Element.Fe: "d",
    Element.Co: "d",
    Element.Ni: "d",
    Element.Cu: "d",
    Element.Zn: "d",
    Element.Ga: "p",
    Element.Ge: "p",
    Element.As: "p",
    Element.Se: "p",
    Element.Br: "p",
    Element.Kr: "p",
    Element.Rb: "s",
    Element.Sr: "s",
    Element.Y: "d",
    Element.Zr: "d",
    Element.Nb: "d",
    Element.Mo: "d",
    Element.Tc: "d",
    Element.Ru: "d",
    Element.Rh: "d",
    Element.Pd: "d",
    Element.Ag: "d",
    Element.Cd: "d",
    Element.In: "p",
    Element.Sn: "p",
    Element.Sb: "p",
    Element.Te: "p",
    Element.I: "p",
    Element.Xe: "p",
    Element.Cs: "s",
    Element.Ba: "s",
    Element.La: "f",
    Element.Ce: "f",
    Element.Pr: "f",
    Element.Nd: "f",
    Element.Pm: "f",
    Element.Sm: "f",
    Element.Eu: "f",
    Element.Gd: "f",
    Element.Tb: "f",
    Element.Dy: "f",
    Element.Ho: "f",
    Element.Er: "f",
    Element.Tm: "f",
    Element.Yb: "f",
    Element.Lu: "d",
    Element.Hf: "d",
    Element.Ta: "d",
    Element.W: "d",
    Element.Re: "d",
    Element.Os: "d",
    Element.Ir: "d",
    Element.Pt: "d",
    Element.Au: "d",
    Element.Hg: "d",
    Element.Tl: "p",
    Element.Pb: "p",
    Element.Bi: "p",
    Element.Po: "p",
    Element.At: "p",
    Element.Rn: "p",
    Element.Fr: "s",
    Element.Ra: "s",
    Element.Ac: "f",
    Element.Th: "f",
    Element.Pa: "f",
    Element.U: "f",
    Element.Np: "f",
    Element.Pu: "f",
    Element.Am: "f",
    Element.Cm: "f",
    Element.Bk: "f",
    Element.Cf: "f",
    Element.Es: "f",
    Element.Fm: "f",
    Element.Md: "f",
    Element.No: "f",
    Element.Lr: "d",
    Element.Rf: "d",
    Element.Db: "d",
    Element.Sg: "d",
    Element.Bh: "d",
    Element.Hs: "d",
    Element.Mt: "d",
    Element.Ds: "d",
    Element.Rg: "d",
    Element.Cn: "d",
    Element.Nh: "p",
    Element.Fl: "p",
    Element.Mc: "p",
    Element.Lv: "p",
    Element.Ts: "p",
    Element.Og: "p",
}
