from wikidataintegrator import wdi_core

def get_wikidata_info(qid):
    my_first_wikidata_item = wdi_core.WDItemEngine(wd_item_id=qid)
    all_info = my_first_wikidata_item.get_wd_json_representation()
    print([item['value'] for item in all_info['aliases']['en']])
    return all_info

def get_wikidata_aliases(qid):
    my_first_wikidata_item = wdi_core.WDItemEngine(wd_item_id=qid)
    all_info = my_first_wikidata_item.get_wd_json_representation()
    aliases = [all_info['labels']['en']['value']]
    for lang in all_info['aliases'].keys():
        if lang == 'en':
            aliases += [item['value'] for item in all_info['aliases'][lang]]
    return aliases