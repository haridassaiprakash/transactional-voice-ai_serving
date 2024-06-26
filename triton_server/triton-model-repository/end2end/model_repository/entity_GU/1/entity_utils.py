import os
import re
import yaml

ENTITY_VARIATION_PATH = "entity-variations/{}.yaml"
ENTITY_PATTERN_PATH = "entity-variations/patterns.yaml"


def intersection_check(start1, end1, start2, end2):
    if (
        start1 <= start2 <= end1
        or start1 <= end2 <= end1
        or start2 <= start1 <= end2
        or start2 <= end1 <= end2
    ):
        return True
    else:
        return False


class EntityRecognizer:
    def __init__(self, lang, base_path):
        self.lang = lang
        with open(os.path.join(base_path, ENTITY_VARIATION_PATH.format(lang)), encoding="utf-8") as f:
            self.variations_dict = yaml.load(f, yaml.BaseLoader)
        with open(os.path.join(base_path, ENTITY_PATTERN_PATH.format(lang)), encoding="utf-8") as f:
            self.pattern_dict = yaml.load(f, yaml.BaseLoader)
        self.patterns = dict()
        for entity_type, pattern_list in self.pattern_dict.items():
            if len(pattern_list) == 1:
                pattern = pattern_list[0]
            else:
                pattern = "|".join(pattern_list)
            self.patterns[entity_type] = pattern

    def create_entity_dict_from_match(self, ent_type, ent_word, ent_val, start, end):
        return {
            "entity": ent_type,
            "word": ent_word,
            "value": ent_val,
            "start": start,
            "end": end,
        }

    def predict(self, sentence, sentence_itn=None):
        if not sentence_itn:
            sentence_itn = sentence
        entities = list()
        for ent_type in self.variations_dict:
            for ent_val, ent_variations in self.variations_dict[ent_type].items():
                for variation in ent_variations:
                    variation = variation.replace(" ", "")
                    variation = variation.lower()
                    variation_pattern = r"{}".format("\s*".join(list(variation)))
                    match = list(re.finditer(variation_pattern, sentence))
                    if not match:
                        continue
                    for m in match:
                        start = m.start()
                        end = m.end()
                        if start > 0 and sentence[start - 1] != " ":
                            continue
                        if end < len(sentence) and sentence[end] != " ":
                            continue
                        entities.append(
                            self.create_entity_dict_from_match(
                                ent_type, variation, ent_val, m.start(), m.end()
                            )
                        )

        for ent_type, pattern in self.patterns.items():
            match = list(re.finditer(pattern, sentence_itn))
            for m in match:
                entities.append(
                    self.create_entity_dict_from_match(
                        ent_type, m.group(0), m.group(0), m.start(), m.end()
                    )
                )

        filtered_entities = self.remove_overlap(entities)
        formatted_entities = self.format_entities(filtered_entities)
        final_entities = self.remove_duplicate(formatted_entities)
        ########### TEMP UGLY FIX ###########################
        final_entities = self.fix_joint_number_amount(final_entities)
        ###################################################
        return final_entities

    def remove_overlap(self, entities):
        entities = sorted(entities, key=lambda x: x["end"] - x["start"], reverse=True)
        filtered_entities = list()
        for i, ent in enumerate(entities):
            intersects = False
            for j, f_ent in enumerate(filtered_entities):
                if intersection_check(
                    ent["start"], ent["end"], f_ent["start"], f_ent["end"]
                ):
                    intersects = True
            if not intersects:
                filtered_entities.append(ent)

        return filtered_entities

    def remove_duplicate(self, entities):
        filtered_entities = list()
        retained_type_val = set()
        for ent in entities:
            type_val = (ent["entity"], ent["value"])
            if type_val in retained_type_val:
                continue
            retained_type_val.add(type_val)
            filtered_entities.append(ent)
        return filtered_entities

    def format_amount(self, ent_val):
        remove_rupees_substring = [
            "rupees",
            "rupee",
            "rs",
            "rupes",
            "ruppes",
            "रूपीस",
            "रुपीस",
            "रूपी",
            "रुपी",
            "रूपये",
            "रुपये",
            "रूपय",
            "रुपय",
            "रूपए",
            "रुपए",
            "ரூபா",
            "ரூபாய்",
            "ரூபாய்க்கு"
            "ரூவா",
            "ருபீஸ்",
            "ருபி",
            "ଟଙ୍କା",
            "ରୂପୀସ",
            "ରୂପୀ",
            "ରୁପିସ",
            "ରୁପି",
            "₹",
            "రూపాయి",
            "రూపాయలు",
            "రూపాయ",
            "ರೂಪಾಯಿ್ದು",
            'ರೂಪಾಯ',
            "ರೂಪಾಯ್ದು",
            "ರೂಪಾಯದ್ದು",
            "ರುಪಾಯ್ದ್",
            "ರೂಪಾಯಿ",
            "ಹಣ",
            "రుపల్",
            "రూపల",
            "ਰੁਪਇਆ",
            "ਰੁਪਆਂ",
            "ਰੁਪਇਆਂ",
            "ਰੁਪਿਆਂ",
            "ਰੁਪਇਆ",
            "টাকার",
            "টাকা",
        ]
        for ss in remove_rupees_substring:
            ent_val = ent_val.replace(ss, "")
        ent_val = re.sub(r'\D', '', ent_val)
        ent_val = ent_val.replace(" ", "")
        ent_val = ent_val.strip()
        return ent_val

    def format_entities(self, entities):
        fn_dict = {
            "amount_of_money": self.format_amount,
            "mobile_number": lambda x: "".join(x.strip().split()),
            "vehicle_number": lambda x: "".join(x.strip().split()),
        }

        entities_formatted = list()
        for ent in entities:
            if ent["entity"] in fn_dict:
                ent["value"] = fn_dict[ent["entity"]](ent["value"])
                entities_formatted.append(ent)
            else:
                entities_formatted.append(ent)
        return entities_formatted

    ############### TEMP UGLY FIX ##############################################
    def fix_joint_number_amount(self, entities):
        """
        Fix the data of format: 9 9 9 9 9 9 9 9 9 9 1 rupee
        where first 10 digits are mobile number and later digits is amount of money
        """
        new_entities = list()
        for ent in entities:
            if ent["entity"] == "amount_of_money" and len(ent["value"]) > 10:
                mobile_val, amount_val = ent["value"][:10], ent["value"][10:]
                mobile_ent = self.create_entity_dict_from_match(
                    "mobile_number", ent["word"], mobile_val, ent["start"], ent["end"]
                )
                amount_ent = self.create_entity_dict_from_match(
                    "amount_of_money", ent["word"], amount_val, ent["start"], ent["end"]
                )
                new_entities.extend([mobile_ent, amount_ent])
            else:
                new_entities.append(ent)

        return new_entities

    ###########################################################################
