"""
orvue_us_inverse.viewer3d.info - names, colours and teaching text for the anatomy viewer.

Lookups go by structure name; unknown names fall back to a title-cased name and a colour chosen from the
structure's tissue labels (anatomy.TISSUES), so new structures added to anatomy.py still display sensibly.
"""

from orvue_us_inverse.simulation.anatomy import CASES  # noqa: E402

COLOURS = {                      # hex colour, opacity
    "gallbladder": ("#97C459", 0.35),
    "duct": ("#639922", 1.0),
    "artery": ("#E24B4A", 1.0),
    "vein": ("#378ADD", 1.0),
    "stone": ("#FAC775", 1.0),
    "node": ("#AFA9EC", 1.0),
    "liver": ("#D85A30", 0.12),
    "fat": ("#F2D16B", 0.10),
    "calot": ("#EF9F27", 0.4),
    "other": ("#B4B2A9", 1.0),
}

# tissue label -> colour class (labels from anatomy.TISSUES)
LABEL_CLASS = {2: "gallbladder", 3: "gallbladder", 4: "stone", 5: "duct", 6: "artery", 7: "artery",
               8: "vein", 9: "vein", 10: "node", 0: "liver", 1: "fat"}

PRETTY = {
    "gallbladder": "Gallbladder",
    "cystic_duct": "Cystic duct",
    "chd_cbd": "Common hepatic duct → common bile duct",
    "proper_hepatic_artery": "Proper hepatic artery",
    "left_hepatic_artery": "Left hepatic artery",
    "right_hepatic_artery": "Right hepatic artery",
    "cystic_artery": "Cystic artery",
    "cystic_artery_superficial": "Cystic artery, superficial branch",
    "cystic_artery_deep": "Cystic artery, deep branch",
    "cystic_branch_1": "Cystic artery branch (from the RHA)",
    "portal_vein": "Portal vein",
    "left_portal_vein": "Left portal vein",
    "calot_lymph_node": "Calot's lymph node",
    "cbd_stone": "Stone in the common bile duct",
    "liver": "Liver",
    "fat": "Fat layer",
    "calot_triangle": "Calot's triangle",
}

DESCRIPTION = {
    "gallbladder": "A pear-shaped bag under the liver that stores and concentrates bile. In a cholecystectomy it "
                   "is the organ being removed, so its neck is where the surgeon must identify what leaves it "
                   "before cutting anything. On ultrasound the bile inside looks black.",
    "cystic_duct": "The narrow tube that drains the gallbladder into the main bile duct. It is one of the two "
                   "structures that are clipped and cut during the operation. Mistaking the main bile duct for "
                   "it is the classic cause of bile duct injury.",
    "chd_cbd": "The main bile duct: the common hepatic duct carries bile down from the liver and becomes the "
               "common bile duct once the cystic duct joins it, emptying into the gut. It must never be cut; "
               "injuring it is the most serious complication of a cholecystectomy.",
    "proper_hepatic_artery": "The main artery bringing oxygen-rich blood to the liver. It divides into the right "
                             "and left hepatic arteries near the liver's hilum. It runs close to the bile duct "
                             "and must be protected.",
    "left_hepatic_artery": "The branch of the hepatic artery that supplies the left part of the liver. It runs "
                           "away from the gallbladder and is rarely at risk, but it helps orient the anatomy.",
    "right_hepatic_artery": "The branch of the hepatic artery that supplies the right part of the liver and "
                            "usually gives off the cystic artery. It often passes close to Calot's triangle and "
                            "can be mistaken for the cystic artery.",
    "cystic_artery": "The small artery that supplies the gallbladder, usually a branch of the right hepatic "
                     "artery inside Calot's triangle. It is the second structure clipped and cut in the "
                     "operation; bleeding from it is a common difficulty.",
    "cystic_artery_superficial": "The branch of the cystic artery running over the free (lower) surface of the "
                                 "gallbladder. It is divided along with the main cystic artery.",
    "cystic_artery_deep": "The branch of the cystic artery running between the gallbladder and the liver bed. It "
                          "can bleed when the gallbladder is peeled off the liver.",
    "cystic_branch_1": "A short cystic artery arising from a looping right hepatic artery (\"caterpillar hump\"). "
                       "The loop lies right against the gallbladder neck, so the right hepatic artery itself can "
                       "be clipped by mistake.",
    "portal_vein": "The large vein that brings blood from the gut to the liver. It lies behind (deeper than) the "
                   "bile duct and hepatic artery. It is not dissected in a cholecystectomy but is a landmark on "
                   "ultrasound: a large dark tube with bright walls.",
    "left_portal_vein": "The branch of the portal vein to the left part of the liver. Useful for orientation near "
                        "the hilum.",
    "calot_lymph_node": "A small lymph node that sits on the cystic artery in Calot's triangle. Surgeons use it "
                        "as a pointer to the artery; it can be enlarged when the gallbladder is inflamed.",
    "cbd_stone": "A gallstone that has slipped into the common bile duct. It can block bile flow (jaundice) and "
                 "must be found before or during surgery; on ultrasound it is bright with a dark shadow below.",
    "liver": "The large organ in front of (above) the gallbladder in this view. The gallbladder sits in a shallow "
             "bed on its underside. The liver's lower edge is where the probe changes from imaging through liver "
             "to imaging through fat.",
    "fat": "A thin layer of fat and connective tissue under the scanning surface. It is thicker in obese "
           "patients, which pushes everything deeper and makes both surgery and ultrasound harder.",
    "calot_triangle": "Calot's (hepatocystic) triangle: the space bounded by the cystic duct, the common hepatic "
                      "duct and the liver. Clearing it until only two structures enter the gallbladder (the "
                      "\"critical view of safety\") is the key step that prevents bile duct injury.",
    "gallstone": "A hard stone formed from bile inside the gallbladder. Stones are the usual reason for the "
                 "operation. On ultrasound they are bright with a black acoustic shadow behind them.",
    "impacted_stone": "A stone stuck in the gallbladder neck (Hartmann's pouch). It keeps the gallbladder "
                      "swollen and inflamed and can press on the bile duct, distorting the anatomy.",
}

CASE_TEXT = {name: text for name, text in CASES.items()}


def colour_class(name, prim=None):
    """Colour class of a structure: by name first, then by its tissue labels."""
    if name == "gallbladder":
        return "gallbladder"
    if name in ("cystic_duct", "chd_cbd"):
        return "duct"
    if "artery" in name or name.startswith("cystic_branch"):
        return "artery"
    if "portal" in name or "vein" in name:
        return "vein"
    if "stone" in name:
        return "stone"
    if "node" in name:
        return "node"
    if name in ("liver", "fat"):
        return name
    if name == "calot_triangle":
        return "calot"
    if prim is not None:
        lab = getattr(prim, "wall_label", None)
        lab = getattr(prim, "label", None) if lab is None else lab
        return LABEL_CLASS.get(lab, "other")
    return "other"


def colour(name, prim=None):
    """(hex colour, opacity) of a structure."""
    return COLOURS[colour_class(name, prim)]


def pretty_name(name):
    if name in PRETTY:
        return PRETTY[name]
    if name.startswith("gallstone_"):
        return f"Gallstone {name.split('_', 1)[1]}"
    if name.startswith("impacted_stone"):
        where = name[len("impacted_stone"):].strip("_").replace("_", " ")
        return f"Impacted stone ({where.title()})" if where else "Impacted stone"
    if name.startswith("cystic_"):
        return "Cystic artery, " + name[len("cystic_"):].replace("_", " ")
    return name.replace("_", " ").strip().capitalize()


def description(name, prim=None):
    if name in DESCRIPTION:
        return DESCRIPTION[name]
    if name.startswith("gallstone"):
        return DESCRIPTION["gallstone"]
    if name.startswith("impacted_stone"):
        return DESCRIPTION["impacted_stone"]
    cls = colour_class(name, prim)
    generic = {"artery": "An artery of the hepatobiliary region in this case.",
               "vein": "A vein of the hepatobiliary region in this case.",
               "duct": "A bile duct in this case.", "stone": DESCRIPTION["gallstone"],
               "node": DESCRIPTION["calot_lymph_node"]}
    return generic.get(cls, "A structure of the simulated anatomy.")


def case_description(case):
    return CASE_TEXT.get(case, "")
