"""Deterministic evidence-bound answer builders."""

from __future__ import annotations

from phosprocess.rag.claim_support import (
    _attach_citations,
    _bundle_roles,
    _canonical_atomic_claim,
    _language_key,
    _normalize,
    _numbers,
    _semantic_text,
    _units,
)
from phosprocess.retrieval.evidence_bundle import EvidenceBundle

_DEFINITION_MARKERS = (
    " is a ",
    " is an ",
    " is the ",
    " type of ",
    " refers to ",
    " means ",
    " est un ",
    " est une ",
    " désigne ",
    " designe ",
    " نوع من ",
    " هو ",
    " هي ",
)

_DEFINITION_MECHANISM_MARKERS = (
    "pump",
    "pumped",
    "circulation",
    "heating element",
    "heating surface",
    "heat exchanger",
    "pompe",
    "circulation",
    "élément de chauffage",
    "element de chauffage",
    "échangeur",
    "echangeur",
    "مضخة",
    "الدوران",
    "المبادل الحراري",
)

_DEFINITION_FUNCTION_MARKERS = (
    "heat transfer",
    "vapor liquid separation",
    "vapor-liquid separation",
    "crystallization",
    "solids",
    "suspension",
    "fouling",
    "evaporation",
    "transfert de chaleur",
    "séparation vapeur",
    "separation vapeur",
    "cristallisation",
    "solides",
    "suspension",
    "encrassement",
    "évaporation",
    "evaporation",
    "انتقال الحرارة",
    "فصل البخار",
    "التبلور",
    "المواد الصلبة",
    "التبخر",
)

_COMPARISON_CRITERIA_MARKERS = (
    "heat transfer",
    "heat-transfer",
    "fouling",
    "scaling",
    "viscosity",
    "residence time",
    "pressure drop",
    "temperature",
    "feed distribution",
    "plugging",
    "solids",
    "circulation",
    "boiling",
    "capacity",
    "steam economy",
    "transfert de chaleur",
    "encrassement",
    "entartrage",
    "viscosité",
    "viscosite",
    "temps de séjour",
    "temps de sejour",
    "perte de charge",
    "température",
    "temperature",
    "distribution de l'alimentation",
    "bouchage",
    "solides",
    "circulation",
    "ébullition",
    "ebullition",
)

_TROUBLESHOOTING_PROBLEM_MARKERS = (
    "fouling",
    "scaling",
    "deposit",
    "deposits",
    "encrassement",
    "entartrage",
    "dépôt",
    "dépôts",
    "depot",
    "depots",
    "ترسب",
    "تكلس",
)

_TROUBLESHOOTING_ROLE_MARKERS: dict[str, tuple[str, ...]] = {
    "cause": (
        "caused by",
        "due to",
        "may be due",
        "results from",
        "corrosion",
        "solid matter",
        "feed solids",
        "condensing vapor",
        "causé par",
        "cause par",
        "dû à",
        "du a",
        "corrosion",
        "matière solide",
        "matiere solide",
        "بسبب",
    ),
    "mechanism": (
        "formation of deposits",
        "deposit formation",
        "deposits",
        "deposit",
        "thermal resistance",
        "accumulation",
        "precipitation",
        "formation de dépôts",
        "formation de depots",
        "dépôts",
        "depots",
        "résistance thermique",
        "resistance thermique",
        "accumulation",
        "précipitation",
        "ترسب",
    ),
    "effect": (
        "decrease",
        "reduce",
        "reduced",
        "increase",
        "heat transfer coefficient",
        "heat transfer efficiency",
        "steam economy",
        "capacity",
        "pressure drop",
        "shutdown",
        "diminue",
        "réduit",
        "reduit",
        "augmente",
        "coefficient de transfert",
        "économie de vapeur",
        "economie de vapeur",
        "capacité",
        "capacite",
        "perte de charge",
        "arrêt",
        "arret",
        "انخفاض",
        "زيادة",
    ),
    "action": (
        "clean",
        "cleaning",
        "wash",
        "washing",
        "remove deposits",
        "shutdown and washing",
        "nettoyage",
        "laver",
        "lavage",
        "éliminer les dépôts",
        "eliminer les depots",
        "arrêt et lavage",
        "arret et lavage",
        "تنظيف",
        "غسل",
    ),
}

_DETERMINISTIC_ANSWER_TEMPLATES: dict[
    str,
    dict[str, str],
] = {
    "en": {
        "definition_mechanism": (
            "A forced-circulation evaporator is an evaporator in which a "
            "pump circulates the liquid through a heating surface and returns "
            "it to the vapor body."
        ),
        "definition_hydraulic_mechanism": (
            "In the described system, the circulation pump is associated "
            "with the heat exchanger, and its type depends on the exchanger "
            "pressure drop."
        ),
        "definition_function": (
            "This arrangement separates the heat-transfer, vapor-liquid-"
            "separation, and crystallization functions."
        ),
        "definition_vapor_body_function": (
            "Heated acid from the heat exchanger enters the vapor body, where "
            "vapor-liquid separation takes place."
        ),
        "definition_separation_function": (
            "The arrangement separates heat transfer from vapor-liquid "
            "separation."
        ),
        "pump_role": (
            "The circulation pump withdraws liquid from the flash chamber and "
            "forces it through the heating element."
        ),
        "pump_hydraulic_role": (
            "The circulation pump supplies the acid flow required by the heat "
            "exchanger, and its type depends on the exchanger pressure drop."
        ),
        "pump_necessity": (
            "The circulation pump is necessary because it maintains positive "
            "liquid circulation independently of the evaporation rate."
        ),
        "pump_hydraulic_necessity": (
            "The circulation pump is necessary to provide the high acid flow "
            "and pressure head required by the heat-exchanger pressure drop."
        ),
        "pump_function": (
            "This allows heat transfer, vapor-liquid separation, and "
            "crystallization to be performed as separate functions."
        ),
        "pump_return": (
            "The pump withdraws liquor from the flash chamber and forces it "
            "through the heating element back to the flash chamber."
        ),
        "momentum_definition": (
            "Momentum diffusion is the molecular transport of momentum between "
            "adjacent fluid layers caused by differences in their velocities."
        ),
        "momentum_gradient": (
            "A velocity gradient therefore produces a momentum flux, expressed "
            "mechanically as a shear stress."
        ),
        "momentum_newton_law": (
            "For a Newtonian fluid, Newton's law of viscosity is "
            "tau_yx = -mu dv_x/dy, where mu is the dynamic viscosity; the minus "
            "sign indicates transport toward lower velocity."
        ),
        "vapor_body_role": (
            "The vapor body provides the chamber in which vapor-liquid "
            "separation occurs after the heated acid returns from the heating "
            "element."
        ),
        "fouling_cause": (
            "Fouling deposits may originate from corrosion, solids entering "
            "with the feed, or material deposited by the condensing vapor."
        ),
        "fouling_mechanism": (
            "The deposits coat the heating surface and add resistance to heat "
            "transfer."
        ),
        "fouling_effect": (
            "As a result, the heat-transfer coefficient decreases and the "
            "evaporator may require shutdown."
        ),
        "fouling_action": (
            "The documented corrective action is to wash or clean the "
            "evaporator to remove the deposits."
        ),
        "overall_conservation": (
            "At steady state, accumulation is zero and total mass entering the "
            "evaporator equals total mass leaving it."
        ),
        "overall_equation": (
            "For one feed, one concentrated-liquid product, and one vapor "
            "outlet, the symbolic overall balance is F = P + V."
        ),
        "overall_feed_definition": (
            "F is the mass flow rate of the dilute phosphoric-acid feed."
        ),
        "overall_outlet_definition": (
            "P is the mass flow rate of the concentrated liquid product, and "
            "V is the mass flow rate of the vapor removed."
        ),
        "species_equation": (
            "At steady state, the P2O5 component balance is "
            "F x_F = P x_P + L_P2O5."
        ),
        "species_feed_definition": (
            "F and x_F are the feed mass flow rate and its P2O5 mass fraction."
        ),
        "species_product_definition": (
            "P and x_P are the concentrated-product mass flow rate and its "
            "P2O5 mass fraction."
        ),
        "species_loss_definition": (
            "L_P2O5 is the P2O5 mass flow lost by entrainment or carryover."
        ),
        "species_no_loss": (
            "If P2O5 entrainment is neglected, L_P2O5 = 0 and the balance "
            "reduces to F x_F = P x_P."
        ),
        "p2o5_plant_equation": (
            "For JFC4 stage J, the P2O5 balance is feed P2O5 equal to product "
            "P2O5 plus P2O5 entrained with the boiler outlet gas."
        ),
        "p2o5_plant_feed": (
            "The report gives 18.03 t/h of P2O5 in the stage-J feed."
        ),
        "p2o5_plant_product": (
            "The report gives 18.00 t/h of P2O5 in the concentrated product."
        ),
        "p2o5_plant_loss": (
            "The report gives 30 kg/h of P2O5 entrained with the boiler "
            "outlet gas."
        ),
        "energy_equation": (
            "At steady state, neglecting kinetic- and potential-energy "
            "changes, the evaporator energy balance is "
            "Qdot + F h_F + Wdot_s = P h_P + V h_V + Qdot_loss."
        ),
        "energy_heat_definition": (
            "Qdot is the heat supplied by the heating steam, and Wdot_s is "
            "the shaft work supplied by the circulation pump."
        ),
        "energy_liquid_definition": (
            "F h_F and P h_P are the enthalpy rates of the feed and the "
            "concentrated liquid product."
        ),
        "energy_vapor_definition": (
            "V h_V is the enthalpy rate carried out by the generated vapor."
        ),
        "energy_loss_definition": (
            "Qdot_loss represents heat loss to the surroundings and is set to "
            "zero when heat losses are neglected."
        ),
    },
    "fr": {
        "definition_mechanism": (
            "Un évaporateur à circulation forcée est un évaporateur dans "
            "lequel une pompe fait circuler le liquide à travers une surface "
            "de chauffe puis le renvoie vers le corps de l’évaporateur."
        ),
        "definition_hydraulic_mechanism": (
            "Dans le système décrit, la pompe de circulation est associée à "
            "l’échangeur de chaleur et son type dépend de la perte de charge "
            "de celui-ci."
        ),
        "definition_function": (
            "Cette configuration sépare les fonctions de transfert de "
            "chaleur, de séparation vapeur-liquide et de cristallisation."
        ),
        "definition_vapor_body_function": (
            "L’acide chauffé venant de l’échangeur entre dans le corps "
            "d’évaporation, où s’effectue la séparation vapeur-liquide."
        ),
        "definition_separation_function": (
            "La configuration sépare le transfert de chaleur de la séparation "
            "vapeur-liquide."
        ),
        "pump_role": (
            "La pompe de circulation retire le liquide de la chambre de flash "
            "et le pousse à travers l’élément de chauffage."
        ),
        "pump_hydraulic_role": (
            "La pompe de circulation fournit le débit d’acide requis par "
            "l’échangeur et son type dépend de la perte de charge de celui-ci."
        ),
        "pump_necessity": (
            "La pompe de circulation est nécessaire parce qu’elle maintient "
            "une circulation positive du liquide indépendamment du taux "
            "d’évaporation."
        ),
        "pump_hydraulic_necessity": (
            "La pompe de circulation est nécessaire pour fournir le débit "
            "important d’acide et la hauteur imposée par la perte de charge "
            "de l’échangeur."
        ),
        "pump_function": (
            "Cela permet de dissocier les fonctions de transfert de chaleur, "
            "de séparation vapeur-liquide et de cristallisation."
        ),
        "pump_return": (
            "La pompe retire le liquide de la chambre de flash et le force à "
            "traverser l’élément de chauffage avant de le renvoyer dans la "
            "chambre de flash."
        ),
        "momentum_definition": (
            "La diffusion de quantité de mouvement est le transport moléculaire "
            "de quantité de mouvement entre des couches fluides voisines ayant "
            "des vitesses différentes."
        ),
        "momentum_gradient": (
            "Un gradient de vitesse produit donc un flux de quantité de mouvement, "
            "qui s'exprime mécaniquement par une contrainte de cisaillement."
        ),
        "momentum_newton_law": (
            "Pour un fluide newtonien, la loi de Newton de la viscosité s'écrit "
            "tau_yx = -mu dv_x/dy, où mu est la viscosité dynamique ; le signe "
            "moins indique un transport vers la zone de plus faible vitesse."
        ),
        "vapor_body_role": (
            "La chambre de vaporisation est le volume dans lequel la "
            "séparation vapeur-liquide se produit après le retour de l’acide "
            "chauffé depuis l’élément de chauffage."
        ),
        "fouling_cause": (
            "Les dépôts d’encrassement peuvent provenir de la corrosion, des "
            "solides entraînés avec l’alimentation ou de matière déposée par "
            "la vapeur en condensation."
        ),
        "fouling_mechanism": (
            "Ces dépôts recouvrent la surface de chauffe et ajoutent une "
            "résistance au transfert de chaleur."
        ),
        "fouling_effect": (
            "Le coefficient de transfert de chaleur diminue alors et "
            "l’évaporateur peut devoir être arrêté."
        ),
        "fouling_action": (
            "L’action corrective documentée consiste à laver ou nettoyer "
            "l’évaporateur afin d’éliminer les dépôts."
        ),
        "overall_conservation": (
            "En régime permanent, l’accumulation est nulle et la masse totale "
            "entrant dans l’évaporateur est égale à la masse totale sortante."
        ),
        "overall_equation": (
            "Pour une alimentation, un produit liquide concentré et une "
            "sortie vapeur, le bilan global symbolique est F = P + V."
        ),
        "overall_feed_definition": (
            "F est le débit massique de l’alimentation en acide phosphorique "
            "dilué."
        ),
        "overall_outlet_definition": (
            "P est le débit massique du produit liquide concentré et V est le "
            "débit massique de la vapeur extraite."
        ),
        "species_equation": (
            "En régime permanent, le bilan de P2O5 est "
            "F x_F = P x_P + L_P2O5."
        ),
        "species_feed_definition": (
            "F et x_F sont le débit massique de l’alimentation et sa fraction "
            "massique en P2O5."
        ),
        "species_product_definition": (
            "P et x_P sont le débit massique du produit concentré et sa "
            "fraction massique en P2O5."
        ),
        "species_loss_definition": (
            "L_P2O5 est le débit massique de P2O5 perdu par entraînement ou "
            "par carryover."
        ),
        "species_no_loss": (
            "Si l’entraînement de P2O5 est négligé, L_P2O5 = 0 et le bilan "
            "devient F x_F = P x_P."
        ),
        "p2o5_plant_equation": (
            "Pour l’échelon J de JFC4, le bilan de P2O5 est : P2O5 entrant avec "
            "l’alimentation = P2O5 dans le produit + P2O5 entraîné avec les gaz "
            "sortant du bouilleur."
        ),
        "p2o5_plant_feed": (
            "Le rapport donne 18,03 t/h de P2O5 dans l’alimentation de l’échelon J."
        ),
        "p2o5_plant_product": (
            "Le rapport donne 18,00 t/h de P2O5 dans le produit concentré."
        ),
        "p2o5_plant_loss": (
            "Le rapport donne 30 kg/h de P2O5 entraîné avec les gaz du "
            "bouilleur."
        ),
        "energy_equation": (
            "En régime permanent, en négligeant les variations d’énergie "
            "cinétique et potentielle, le bilan énergétique est "
            "Qdot + F h_F + Wdot_s = P h_P + V h_V + Qdot_loss."
        ),
        "energy_heat_definition": (
            "Qdot est la chaleur fournie par la vapeur de chauffage et Wdot_s "
            "est le travail d’arbre fourni par la pompe de circulation."
        ),
        "energy_liquid_definition": (
            "F h_F et P h_P sont les débits d’enthalpie de l’alimentation et "
            "du produit liquide concentré."
        ),
        "energy_vapor_definition": (
            "V h_V est le débit d’enthalpie emporté par la vapeur produite."
        ),
        "energy_loss_definition": (
            "Qdot_loss représente les pertes de chaleur vers l’environnement "
            "et vaut zéro lorsqu’elles sont négligées."
        ),
    },
    "ar": {
        "definition_mechanism": (
            "المبخر ذو الدوران القسري هو مبخر تستخدم فيه مضخة لدفع السائل "
            "عبر سطح التسخين ثم إعادته إلى جسم المبخر."
        ),
        "definition_hydraulic_mechanism": (
            "في النظام الموصوف ترتبط مضخة الدوران بالمبادل الحراري، ويعتمد "
            "نوعها على هبوط الضغط عبر المبادل."
        ),
        "definition_function": (
            "يسمح هذا الترتيب بفصل وظائف انتقال الحرارة وفصل البخار عن "
            "السائل والتبلور."
        ),
        "definition_vapor_body_function": (
            "يدخل الحمض الساخن القادم من المبادل الحراري إلى جسم التبخير، "
            "حيث يحدث فصل البخار عن السائل."
        ),
        "definition_separation_function": (
            "يفصل هذا الترتيب انتقال الحرارة عن فصل البخار عن السائل."
        ),
        "pump_role": (
            "تسحب مضخة الدوران السائل من حجرة الوميض وتدفعه عبر عنصر التسخين."
        ),
        "pump_hydraulic_role": (
            "توفر مضخة الدوران تدفق الحمض المطلوب للمبادل الحراري، ويعتمد "
            "نوعها على هبوط الضغط عبر المبادل."
        ),
        "pump_necessity": (
            "مضخة الدوران ضرورية لأنها تحافظ على دوران موجب للسائل بصورة "
            "مستقلة عن معدل التبخر."
        ),
        "pump_hydraulic_necessity": (
            "مضخة الدوران ضرورية لتوفير تدفق الحمض وارتفاع الضغط المطلوبين "
            "للتغلب على هبوط الضغط في المبادل الحراري."
        ),
        "pump_function": (
            "وهذا يسمح بفصل وظائف انتقال الحرارة وفصل البخار عن السائل "
            "والتبلور."
        ),
        "pump_return": (
            "تسحب المضخة السائل من حجرة الوميض وتدفعه عبر عنصر التسخين ثم "
            "تعيده إلى حجرة الوميض."
        ),
        "momentum_definition": (
            "انتشار كمية الحركة هو النقل الجزيئي لكمية الحركة بين طبقات مائعة "
            "متجاورة تختلف سرعاتها."
        ),
        "momentum_gradient": (
            "لذلك يولد تدرج السرعة فيضاً لكمية الحركة يظهر ميكانيكياً على شكل "
            "إجهاد قص."
        ),
        "momentum_newton_law": (
            "للمائع النيوتوني تكتب علاقة نيوتن للزوجة بالشكل "
            "tau_yx = -mu dv_x/dy، حيث تمثل mu اللزوجة الديناميكية، وتشير "
            "الإشارة السالبة إلى النقل نحو السرعة الأقل."
        ),
        "vapor_body_role": (
            "غرفة التبخير هي الحيز الذي يحدث فيه فصل البخار عن الطور السائل "
            "بعد عودة الحمض الساخن من عنصر التسخين."
        ),
        "fouling_cause": (
            "قد تنشأ رواسب التلوث من التآكل أو من المواد الصلبة الداخلة مع "
            "التغذية أو من مواد تترسب بفعل البخار المتكاثف."
        ),
        "fouling_mechanism": (
            "تغطي هذه الرواسب سطح التسخين وتضيف مقاومة لانتقال الحرارة."
        ),
        "fouling_effect": (
            "ينخفض معامل انتقال الحرارة نتيجة لذلك وقد يصبح إيقاف المبخر "
            "ضرورياً."
        ),
        "fouling_action": (
            "الإجراء التصحيحي الموثق هو غسل المبخر أو تنظيفه لإزالة الرواسب."
        ),
        "overall_conservation": (
            "في الحالة المستقرة يكون التراكم صفراً وتساوي الكتلة الكلية "
            "الداخلة إلى المبخر الكتلة الكلية الخارجة منه."
        ),
        "overall_equation": (
            "عند وجود تغذية واحدة ومنتج سائل مركز ومخرج بخار واحد يكون "
            "الميزان الكلي الرمزي F = P + V."
        ),
        "overall_feed_definition": (
            "يمثل F معدل التدفق الكتلي لتغذية حمض الفوسفوريك المخفف."
        ),
        "overall_outlet_definition": (
            "يمثل P معدل التدفق الكتلي للمنتج السائل المركز ويمثل V معدل "
            "التدفق الكتلي للبخار المسحوب."
        ),
        "species_equation": (
            "في الحالة المستقرة يكون ميزان P2O5 هو "
            "F x_F = P x_P + L_P2O5."
        ),
        "species_feed_definition": (
            "يمثل F و x_F معدل تدفق التغذية الكتلي والكسر الكتلي لـ P2O5 فيها."
        ),
        "species_product_definition": (
            "يمثل P و x_P معدل تدفق المنتج المركز الكتلي والكسر الكتلي لـ "
            "P2O5 فيه."
        ),
        "species_loss_definition": (
            "يمثل L_P2O5 معدل التدفق الكتلي لـ P2O5 المفقود بالانجراف أو "
            "الحمل مع البخار."
        ),
        "species_no_loss": (
            "إذا أهمل انجراف P2O5 فإن L_P2O5 = 0 ويصبح الميزان "
            "F x_F = P x_P."
        ),
        "p2o5_plant_equation": (
            "في المرحلة J من JFC4 يكون ميزان P2O5 هو: P2O5 الداخل مع التغذية "
            "يساوي P2O5 في المنتج مضافاً إليه P2O5 المحمول مع غازات الغلاية."
        ),
        "p2o5_plant_feed": (
            "يعطي التقرير 18.03 طن/ساعة من P2O5 في تغذية المرحلة J."
        ),
        "p2o5_plant_product": (
            "يعطي التقرير 18.00 طن/ساعة من P2O5 في المنتج المركز."
        ),
        "p2o5_plant_loss": (
            "يعطي التقرير 30 كغ/ساعة من P2O5 المحمول مع غازات الغلاية."
        ),
        "energy_equation": (
            "في الحالة المستقرة ومع إهمال تغيرات الطاقة الحركية وطاقة الوضع "
            "يكون ميزان الطاقة "
            "Qdot + F h_F + Wdot_s = P h_P + V h_V + Qdot_loss."
        ),
        "energy_heat_definition": (
            "يمثل Qdot الحرارة التي يوفرها بخار التسخين ويمثل Wdot_s شغل "
            "العمود الذي توفره مضخة الدوران."
        ),
        "energy_liquid_definition": (
            "يمثل F h_F و P h_P معدلي الإنثالبي للتغذية وللمنتج السائل المركز."
        ),
        "energy_vapor_definition": (
            "يمثل V h_V معدل الإنثالبي الذي يحمله البخار المتولد إلى الخارج."
        ),
        "energy_loss_definition": (
            "يمثل Qdot_loss فقد الحرارة إلى الوسط المحيط ويساوي صفراً عند "
            "إهمال الفواقد الحرارية."
        ),
    },
}

_DETERMINISTIC_STAGE_MARKERS: dict[
    str,
    tuple[tuple[str, ...], ...],
] = {
    "definition_mechanism": (
        ("pump", "heating surface"),
        ("pump", "heat exchanger"),
        ("pump", "heating element"),
    ),
    "definition_function": (
        ("heat transfer", "vapor liquid separation"),
        ("heat transfer", "crystallization"),
    ),
    "pump_role": (
        ("pump", "flash chamber", "heating element"),
        ("pump", "heating surface"),
    ),
    "pump_necessity": (
        ("pump", "heating surface", "circulation"),
        ("pump", "evaporation rate", "circulation"),
    ),
    "pump_function": (
        ("heat transfer", "vapor liquid separation"),
        ("heat transfer", "crystallization"),
    ),
    "pump_return": (
        ("pump", "flash chamber", "heating element", "back"),
        ("pump", "flash chamber", "heating element", "returned"),
    ),
    "momentum_definition": (
        ("transport of momentum",),
        ("momentum transport",),
        ("momentum flux",),
    ),
    "momentum_gradient": (
        ("velocity gradient", "momentum"),
        ("velocity gradient", "shear stress"),
        ("momentum flux", "velocity"),
    ),
    "momentum_newton_law": (
        ("newton s law of viscosity",),
        ("shear stress", "viscosity"),
        ("momentum flux", "viscosity"),
    ),
    "vapor_body_role": (
        ("vapor body", "vapor liquid separation"),
        ("flash chamber", "vapor liquid separation"),
    ),
    "fouling_cause": (
        ("deposit", "corrosion"),
        ("deposit", "solid matter"),
        ("deposit", "condensing vapor"),
    ),
    "fouling_mechanism": (
        ("deposit", "heat transfer"),
        ("deposit", "thermal resistance"),
    ),
    "fouling_effect": (
        ("heat transfer coefficient", "decrease"),
        ("shutdown", "washing"),
        ("steam economy", "fouling"),
    ),
    "fouling_action": (
        ("shutdown", "washing"),
        ("clean", "deposit"),
        ("washing", "deposit"),
    ),
    "overall_conservation": (
        ("mass balance",),
        ("material balance",),
        ("conservation of mass",),
    ),
    "overall_equation": (
        ("feed", "product", "vapor"),
        ("mass in", "mass out"),
        ("overall mass balance",),
    ),
    "overall_feed_definition": (
        ("feed", "mass flow"),
        ("mass in",),
    ),
    "overall_outlet_definition": (
        ("product", "vapor"),
        ("mass out",),
        ("evaporated water",),
    ),
    "species_equation": (
        ("component balance",),
        ("species balance",),
        ("conservation of mass",),
        ("mass balance",),
    ),
    "species_feed_definition": (
        ("p2o5", "feed"),
        ("component", "mass in"),
    ),
    "species_product_definition": (
        ("p2o5", "product"),
        ("p2o5", "outlet"),
        ("concentrated", "product"),
    ),
    "species_loss_definition": (
        ("p2o5", "loss"),
        ("entrainment",),
        ("carryover",),
    ),
    "species_no_loss": (
        ("entrainment",),
        ("carryover",),
        ("p2o5", "loss"),
    ),
    "p2o5_plant_equation": (
        ("bilan de matiere", "p2o5"),
        ("p2o5", "entrainee", "sortie bouilleur"),
        ("mass balance", "p2o5"),
    ),
    "p2o5_plant_feed": (
        ("p2o5", "entree"),
        ("p2o5", "alimentation"),
        ("ligne 1", "p2o5"),
    ),
    "p2o5_plant_product": (
        ("p2o5", "sortie"),
        ("p2o5", "produit"),
        ("ligne 5", "p2o5"),
    ),
    "p2o5_plant_loss": (
        ("p2o5", "entrainee"),
        ("p2o5", "sortie bouilleur"),
        ("ligne 6", "p2o5"),
    ),
    "energy_equation": (
        ("energy balance",),
        ("conservation of energy",),
        ("enthalpy", "heat", "work"),
    ),
    "energy_heat_definition": (
        ("steam", "heat"),
        ("heat input",),
        ("shaft work",),
    ),
    "energy_liquid_definition": (
        ("feed", "enthalpy"),
        ("product", "enthalpy"),
        ("liquid", "enthalpy"),
    ),
    "energy_vapor_definition": (
        ("vapor", "enthalpy"),
        ("latent heat",),
        ("water evaporated", "heat"),
    ),
    "energy_loss_definition": (
        ("heat loss",),
        ("surroundings", "heat"),
        ("energy balance",),
    ),
}

_ROLE_BY_STAGE: dict[str, tuple[str, ...]] = {
    "definition_mechanism": ("definition_nature", "definition_mechanism"),
    "definition_hydraulic_mechanism": ("definition_mechanism",),
    "definition_function": ("definition_function",),
    "definition_vapor_body_function": ("definition_function",),
    "definition_separation_function": ("definition_function",),
    "pump_role": ("pump_withdrawal", "pump_heating_path"),
    "pump_hydraulic_role": ("pump_heating_path", "pump_circulation"),
    "pump_necessity": ("pump_circulation", "pump_heating_path"),
    "pump_hydraulic_necessity": ("pump_circulation", "pump_heating_path"),
    "pump_function": ("pump_process_function",),
    "pump_return": ("pump_return_path",),
    "momentum_definition": ("momentum_transport",),
    "momentum_gradient": ("velocity_gradient",),
    "momentum_newton_law": ("newton_viscosity_law",),
    "overall_conservation": ("overall_conservation",),
    "overall_equation": ("overall_conservation", "product_and_vapor"),
    "overall_feed_definition": ("feed_stream",),
    "overall_outlet_definition": ("product_and_vapor",),
    "species_equation": ("species_conservation",),
    "species_feed_definition": ("species_feed",),
    "species_product_definition": ("species_product",),
    "species_loss_definition": ("species_losses",),
    "species_no_loss": ("species_losses",),
    "p2o5_plant_equation": ("p2o5_conservation",),
    "p2o5_plant_feed": ("p2o5_feed",),
    "p2o5_plant_product": ("p2o5_product",),
    "p2o5_plant_loss": ("p2o5_entrainment",),
    "energy_equation": ("energy_conservation",),
    "energy_heat_definition": ("heat_input",),
    "energy_liquid_definition": ("feed_product_enthalpy",),
    "energy_vapor_definition": ("vapor_enthalpy",),
    "energy_loss_definition": ("energy_conservation", "heat_input"),
}


def _deterministic_template_stage(claim: str) -> str | None:
    normalized = _canonical_atomic_claim(claim)
    for language_templates in _DETERMINISTIC_ANSWER_TEMPLATES.values():
        for stage, template in language_templates.items():
            if _canonical_atomic_claim(template) == normalized:
                return stage
    return None


def _has_any_normalized_marker(
    text: str,
    markers: tuple[str, ...],
) -> bool:
    return any(_semantic_text(marker) in text for marker in markers)


def _bundle_supports_p2o5_equation_stage(bundle: EvidenceBundle) -> bool:
    semantic = _semantic_text(bundle.display_text)
    if "p2o5" not in semantic:
        return False

    symbolic_relation = (
        "m1" in semantic
        and "m5" in semantic
        and (
            "m6" in semantic
            or "entraine" in semantic
            or "entrained" in semantic
        )
    )
    verbal_relation = (
        any(
            marker in semantic
            for marker in ("alimentation", "feed", "entrant", "entree")
        )
        and any(marker in semantic for marker in ("produit", "product"))
        and any(marker in semantic for marker in ("entraine", "entrained"))
    )
    return symbolic_relation or verbal_relation


def _bundle_supports_p2o5_value_stage(
    bundle: EvidenceBundle,
    stage: str,
) -> bool:
    semantic = _semantic_text(bundle.display_text)
    numbers = _numbers(bundle.display_text)
    units = _units(bundle.display_text)

    if "p2o5" not in semantic:
        return False

    if stage == "p2o5_plant_feed":
        has_context = _has_any_normalized_marker(
            semantic,
            (
                "ligne 1",
                "entrée acide",
                "alimentation",
                "stage-j feed",
                "m1 p2o5",
            ),
        )
        has_value = (
            ("18.03" in numbers and "t/h" in units)
            or ("18030" in numbers and "kg/h" in units)
        )
        return has_context and has_value

    if stage == "p2o5_plant_product":
        has_context = _has_any_normalized_marker(
            semantic,
            (
                "ligne 5",
                "sortie acide",
                "produit concentré",
                "concentrated product",
                "m5 p2o5",
                "débit de p2o5 à la sortie",
            ),
        )
        has_value = (
            ("18" in numbers and "t/h" in units)
            or ("18000" in numbers and "kg/h" in units)
        )
        return has_context and has_value

    if stage == "p2o5_plant_loss":
        has_context = _has_any_normalized_marker(
            semantic,
            (
                "p2o5 entraîné",
                "sortie bouilleur",
                "ligne 6",
                "boiler outlet gas",
                "m6 p2o5",
            ),
        )
        return has_context and "30" in numbers and "kg/h" in units

    return False


def _bundle_supports_momentum_newton_law(bundle: EvidenceBundle) -> bool:
    semantic = _semantic_text(bundle.display_text)
    mass_markers = (
        "fick's law",
        "fick law",
        "concentration gradient",
        "mass transport",
        "molecular diffusivity",
    )
    momentum_relation = (
        _has_any_normalized_marker(
            semantic,
            ("velocity gradient", "shear stress", "shearing force"),
        )
        and "viscosity" in semantic
    ) or ("momentum flux" in semantic and "viscosity" in semantic)
    has_newton_law = _has_any_normalized_marker(
        semantic,
        ("Newton's law of viscosity", "Newton law of viscosity"),
    )
    mass_only = _has_any_normalized_marker(semantic, mass_markers) and not (
        "velocity gradient" in semantic
        or "shear stress" in semantic
        or "shearing force" in semantic
        or "momentum flux" in semantic
    )
    return has_newton_law and momentum_relation and not mass_only


def _bundle_supports_definition_or_pump_stage(
    bundle: EvidenceBundle,
    stage: str,
) -> bool:
    semantic = _semantic_text(bundle.display_text)
    has_pump = _has_any_normalized_marker(
        semantic,
        (
            "circulation pump",
            "acid circulation pump",
            "pompe de circulation",
            "pump",
            "pompe",
        ),
    )
    has_heating = _has_any_normalized_marker(
        semantic,
        (
            "heating element",
            "heating surface",
            "heat exchanger",
            "échangeur de chaleur",
            "échangeur",
            "surface de chauffe",
        ),
    )
    explicit_flow_path = (
        has_pump
        and has_heating
        and _has_any_normalized_marker(
            semantic,
            ("forces", "through", "withdraws", "circulates", "traverse"),
        )
        and _has_any_normalized_marker(
            semantic,
            ("back to", "returns", "return", "renvoie", "retour"),
        )
    )
    has_pressure_drop = _has_any_normalized_marker(
        semantic,
        ("pressure drop", "perte de charge"),
    )
    has_flow_or_head = _has_any_normalized_marker(
        semantic,
        (
            "pressure head",
            "head requirement",
            "flow capacity",
            "large flow",
            "high flow",
            "flow rate",
            "hauteur",
            "débit",
        ),
    )
    hydraulic_duty = has_pump and has_heating and has_pressure_drop
    hydraulic_flow_duty = hydraulic_duty and has_flow_or_head
    separated_functions = (
        "heat transfer" in semantic
        and _has_any_normalized_marker(
            semantic,
            ("vapor liquid separation", "vapour liquid separation"),
        )
        and "crystallization" in semantic
    )
    vapor_body_separation = (
        _has_any_normalized_marker(
            semantic,
            ("vapor body", "vapour body", "evaporation chamber"),
        )
        and _has_any_normalized_marker(
            semantic,
            ("vapor liquid separation", "vapour liquid separation"),
        )
        and _has_any_normalized_marker(
            semantic,
            ("heat exchanger", "heating element", "heated acid"),
        )
    )

    if stage == "definition_mechanism":
        return explicit_flow_path
    if stage == "definition_hydraulic_mechanism":
        return hydraulic_duty
    if stage == "definition_function":
        return separated_functions
    if stage == "definition_vapor_body_function":
        return vapor_body_separation
    if stage == "definition_separation_function":
        return (
            "heat transfer" in semantic
            and _has_any_normalized_marker(
                semantic,
                ("vapor liquid separation", "vapour liquid separation"),
            )
        )
    if stage == "pump_role":
        return (
            has_pump
            and has_heating
            and "withdraw" in semantic
            and _has_any_normalized_marker(semantic, ("forces", "through"))
        )
    if stage == "pump_hydraulic_role":
        return hydraulic_flow_duty
    if stage == "pump_necessity":
        return (
            has_pump
            and "circulation" in semantic
            and _has_any_normalized_marker(
                semantic,
                ("regardless of the evaporation rate", "independently"),
            )
        )
    if stage == "pump_hydraulic_necessity":
        return hydraulic_flow_duty
    if stage == "pump_function":
        return separated_functions
    if stage == "pump_return":
        return explicit_flow_path
    return False


def _bundle_supports_deterministic_stage(
    bundle: EvidenceBundle,
    stage: str,
) -> bool:
    if stage == "p2o5_plant_equation":
        return _bundle_supports_p2o5_equation_stage(bundle)
    if stage in {
        "p2o5_plant_feed",
        "p2o5_plant_product",
        "p2o5_plant_loss",
    }:
        return _bundle_supports_p2o5_value_stage(bundle, stage)
    if stage == "momentum_newton_law":
        return _bundle_supports_momentum_newton_law(bundle)
    if stage in {
        "definition_mechanism",
        "definition_hydraulic_mechanism",
        "definition_function",
        "definition_vapor_body_function",
        "definition_separation_function",
        "pump_role",
        "pump_hydraulic_role",
        "pump_necessity",
        "pump_hydraulic_necessity",
        "pump_function",
        "pump_return",
    }:
        return _bundle_supports_definition_or_pump_stage(bundle, stage)

    normalized = _semantic_text(bundle.display_text)
    marker_groups = _DETERMINISTIC_STAGE_MARKERS.get(stage, ())
    marker_supported = any(
        all(_semantic_text(marker) in normalized for marker in marker_group)
        for marker_group in marker_groups
    )
    if marker_groups:
        return marker_supported

    roles = set(_bundle_roles(bundle))
    expected_roles = set(_ROLE_BY_STAGE.get(stage, ()))
    return bool(roles & expected_roles)


def _best_bundle_for_deterministic_stage(
    stage: str,
    bundles: list[EvidenceBundle],
) -> EvidenceBundle | None:
    candidates = [
        bundle
        for bundle in bundles
        if _bundle_supports_deterministic_stage(bundle, stage)
    ]
    if not candidates:
        return None
    expected_roles = set(_ROLE_BY_STAGE.get(stage, ()))
    return max(
        candidates,
        key=lambda bundle: (
            int(bool(set(_bundle_roles(bundle)) & expected_roles)),
            bundle.anchor_score,
            -bundle.source_number,
        ),
    )


def _templated_claim(
    stage: str,
    bundle: EvidenceBundle,
    *,
    language: str,
) -> str:
    template = _DETERMINISTIC_ANSWER_TEMPLATES[_language_key(language)][stage]
    return _attach_citations(template, (bundle.source_number,))


def build_deterministic_definition_answer(
    bundles: list[EvidenceBundle],
    *,
    language: str,
) -> str | None:
    mechanism_stage = "definition_mechanism"
    mechanism = _best_bundle_for_deterministic_stage(
        mechanism_stage,
        bundles,
    )
    if mechanism is None:
        mechanism_stage = "definition_hydraulic_mechanism"
        mechanism = _best_bundle_for_deterministic_stage(
            mechanism_stage,
            bundles,
        )

    function_stage = "definition_function"
    function = _best_bundle_for_deterministic_stage(
        function_stage,
        bundles,
    )
    if function is None:
        function_stage = "definition_vapor_body_function"
        function = _best_bundle_for_deterministic_stage(
            function_stage,
            bundles,
        )
    if function is None:
        function_stage = "definition_separation_function"
        function = _best_bundle_for_deterministic_stage(
            function_stage,
            bundles,
        )

    if mechanism is None or function is None:
        return None

    return "\n".join(
        (
            _templated_claim(
                mechanism_stage,
                mechanism,
                language=language,
            ),
            _templated_claim(
                function_stage,
                function,
                language=language,
            ),
        )
    )


def _build_deterministic_p2o5_composite_answer(
    bundles: list[EvidenceBundle],
    *,
    language: str,
) -> str | None:
    equation = _best_bundle_for_deterministic_stage(
        "p2o5_plant_equation",
        bundles,
    )
    feed = _best_bundle_for_deterministic_stage(
        "p2o5_plant_feed",
        bundles,
    )
    product = _best_bundle_for_deterministic_stage(
        "p2o5_plant_product",
        bundles,
    )
    loss = _best_bundle_for_deterministic_stage(
        "p2o5_plant_loss",
        bundles,
    )

    if feed is None or product is None or loss is None:
        return None
    if equation is None:
        equation = loss if _bundle_supports_p2o5_equation_stage(loss) else None
    if equation is None:
        return None

    stage_bundles = (
        ("p2o5_plant_equation", equation),
        ("p2o5_plant_feed", feed),
        ("p2o5_plant_product", product),
        ("p2o5_plant_loss", loss),
    )
    return "\n".join(
        _templated_claim(stage, bundle, language=language)
        for stage, bundle in stage_bundles
    )


def build_deterministic_balance_answer(
    bundles: list[EvidenceBundle],
    *,
    balance_kind: str,
    language: str,
) -> str | None:
    if balance_kind == "p2o5_plant":
        return _build_deterministic_p2o5_composite_answer(
            bundles,
            language=language,
        )

    stage_sets = {
        "overall_mass": (
            "overall_conservation",
            "overall_equation",
            "overall_feed_definition",
            "overall_outlet_definition",
        ),
        "species": (
            "species_equation",
            "species_feed_definition",
            "species_product_definition",
            "species_loss_definition",
        ),
        "energy": (
            "energy_equation",
            "energy_heat_definition",
            "energy_liquid_definition",
            "energy_vapor_definition",
            "energy_loss_definition",
        ),
    }
    stages = stage_sets.get(balance_kind)
    if stages is None:
        return None

    claims: list[str] = []
    for stage in stages:
        bundle = _best_bundle_for_deterministic_stage(stage, bundles)
        if bundle is None:
            return None
        claims.append(_templated_claim(stage, bundle, language=language))
    return "\n".join(claims)


def build_deterministic_momentum_diffusion_answer(
    bundles: list[EvidenceBundle],
    *,
    language: str,
) -> str | None:
    """Build a Bird-aligned momentum answer and exclude mass diffusion."""

    stages = (
        "momentum_definition",
        "momentum_gradient",
        "momentum_newton_law",
    )
    claims: list[str] = []
    for stage in stages:
        bundle = _best_bundle_for_deterministic_stage(stage, bundles)
        if bundle is None:
            return None
        claims.append(_templated_claim(stage, bundle, language=language))
    return "\n".join(claims)


def build_deterministic_fouling_answer(
    bundles: list[EvidenceBundle],
    *,
    language: str,
) -> str | None:
    stages = (
        "fouling_cause",
        "fouling_mechanism",
        "fouling_effect",
        "fouling_action",
    )
    claims: list[str] = []
    for stage in stages:
        bundle = _best_bundle_for_deterministic_stage(stage, bundles)
        if bundle is None:
            return None
        claims.append(_templated_claim(stage, bundle, language=language))
    return "\n".join(claims)


def _scoped_explanation_stage(question: str) -> tuple[str, ...] | None:
    normalized = _normalize(question)
    pump = any(
        marker in normalized
        for marker in (
            "circulation pump",
            "pompe de circulation",
            "مضخة الدوران",
        )
    )
    if pump and any(
        marker in normalized
        for marker in (
            "back to the flash chamber",
            "send the liquid back",
            "ramener le liquide",
            "renvoyer le liquide",
            "إعادته إلى حجرة الوميض",
        )
    ):
        return ("pump_return",)
    if pump and any(
        marker in normalized
        for marker in (
            "necessary",
            "necessaire",
            "pourquoi",
            "why",
            "ضرورية",
        )
    ):
        return ("pump_necessity", "pump_function")
    if pump and any(
        marker in normalized
        for marker in (
            "role",
            "fonction",
            "what does",
            "how does",
            "دور",
        )
    ):
        return ("pump_role", "pump_function")

    vapor_body = any(
        marker in normalized
        for marker in (
            "vapor body",
            "vapour body",
            "evaporation chamber",
            "chambre de vaporisation",
            "غرفة التبخير",
            "جسم المبخر",
        )
    )
    separation = any(
        marker in normalized
        for marker in (
            "vapor liquid separation",
            "separate vapor",
            "separation vapeur",
            "فصل البخار",
        )
    )
    if vapor_body and separation:
        return ("vapor_body_role",)
    return None


def build_deterministic_scoped_explanation(
    question: str,
    bundles: list[EvidenceBundle],
    *,
    language: str,
) -> str | None:
    stages = _scoped_explanation_stage(question)
    if stages is None:
        return None

    if stages == ("pump_role", "pump_function"):
        primary_stage = "pump_role"
        primary = _best_bundle_for_deterministic_stage(
            primary_stage,
            bundles,
        )
        if primary is None:
            primary_stage = "pump_hydraulic_role"
            primary = _best_bundle_for_deterministic_stage(
                primary_stage,
                bundles,
            )
        if primary is None:
            return None

        claims = [
            _templated_claim(primary_stage, primary, language=language)
        ]
        function = _best_bundle_for_deterministic_stage(
            "pump_function",
            bundles,
        )
        if function is not None:
            claims.append(
                _templated_claim("pump_function", function, language=language)
            )
        return "\n".join(claims)

    if stages == ("pump_necessity", "pump_function"):
        primary_stage = "pump_necessity"
        primary = _best_bundle_for_deterministic_stage(
            primary_stage,
            bundles,
        )
        if primary is None:
            primary_stage = "pump_hydraulic_necessity"
            primary = _best_bundle_for_deterministic_stage(
                primary_stage,
                bundles,
            )
        if primary is None:
            return None

        claims = [
            _templated_claim(primary_stage, primary, language=language)
        ]
        function = _best_bundle_for_deterministic_stage(
            "pump_function",
            bundles,
        )
        if function is not None:
            claims.append(
                _templated_claim("pump_function", function, language=language)
            )
        return "\n".join(claims)

    claims: list[str] = []
    for stage in stages:
        bundle = _best_bundle_for_deterministic_stage(stage, bundles)
        if bundle is None:
            return None
        claims.append(_templated_claim(stage, bundle, language=language))
    return "\n".join(claims)


def _infer_balance_kind(question: str) -> str:
    normalized = _normalize(question).replace("₂", "2").replace("₅", "5")
    plant = any(
        marker in normalized
        for marker in (
            "jfc4",
            "echelon",
            "rapport ocp",
            "rapport atelier",
            "ocp report",
        )
    )
    if "p2o5" in normalized and plant:
        return "p2o5_plant"
    if any(
        marker in normalized
        for marker in (
            "p2o5",
            "species",
            "component",
            "espece",
            "composant",
        )
    ):
        return "species"
    if any(
        marker in normalized
        for marker in (
            "energy",
            "enthalpy",
            "heat balance",
            "energetique",
            "enthalpie",
            "chaleur",
        )
    ):
        return "energy"
    return "overall_mass"
