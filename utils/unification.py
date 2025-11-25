import pandas as pd


def unify_verification_data(df: pd.DataFrame) -> pd.DataFrame:
    """Expects a dataframe with "species" column."""
    df.species = df.species.replace(['Empty', 'vehicle'], 'empty')
    df.species = df.species.replace('Undefined', 'other')
    df.species = df.species.replace('reddeer', 'red deer')
    df.species = df.species.replace('roedeer', 'roe deer')
    df.species = df.species.replace('wildboar', 'wild boar')
    # df.species = df.species.replace(
    # ['marten', 'weasel', 'stoat', 'mink', 'polecat'], 'mustelid')
    return df


def unify_deepfaune_results(df: pd.DataFrame) -> pd.DataFrame:
    """Expects a dataframe with "detected_animal" column."""
    df.detected_animal = df.detected_animal.replace('lagomorph', 'hare')
    df.detected_animal = df.detected_animal.replace(
        'golden jackal', 'wolf')
    df.detected_animal = df.detected_animal.replace(
        ['nutria', 'marmot'],
        'beaver'
    )
    df.detected_animal = df.detected_animal.replace(
        ['cow', 'chamois', 'equid', 'goat', 'ibex', 'micromammal', 'mouflon'],
        'other'
    )
    df.detected_animal = df.detected_animal.replace('reindeer', 'red deer')
    return df


def unify_speciesnet_results(df: pd.DataFrame) -> pd.DataFrame:
    """Expects a dataframe with "detected_animal" column."""
    # birds
    df.detected_animal = df.detected_animal.replace([
        'jay', 'blue jay', 'owl', 'common woodpigeon', 'common blackbird',
        'great blue heron', 'american robin', 'golden eagle',
        'eurasian buzzard', 'great black hawk', 'black-billed magpie',
        'mourning dove', 'caprimulgidae family', 'mallard', 'horned lark',
        'blood pheasant', 'american crow', 'hermit thrush',
        'pel\'s fishing-owl', 'pileated woodpecker', 'white stork',
        'southern caracara', 'black woodpecker', 'sandhill crane',
        'palawan peacock-pheasant', 'canada goose', 'bald eagle',
        'gruiformes order', 'red-tailed hawk', 'great egret', 'common crane',
        'wood duck', 'ring-necked pheasant', 'european robin', 'wild turkey'],
        'bird'
    )
    # other
    df.detected_animal = df.detected_animal.replace(
        ['snowshoe hare', 'european hare', 'eastern cottontail'], 'hare')
    df.detected_animal = df.detected_animal.replace(
        ['eastern fox squirrel', 'eurasian red squirrel',
         'eastern gray squirrel', 'red squirrel', 'sciurus species'],
        'squirrel')
    df.detected_animal = df.detected_animal.replace(
        ['garden dormouse', 'peromyscus species'], 'rodent')  # gryzoń
    df.detected_animal = df.detected_animal.replace(
        ['martes species', 'pine marten', 'fisher', 'american marten',
         'beech marten'],
        'marten',  # kuna
    )
    df.detected_animal = df.detected_animal.replace(
        ['siberian weasel', 'weasel family'], 'weasel')  # łasica
    df.detected_animal = df.detected_animal.replace(
        [], 'stoat')  # gronostaj
    df.detected_animal = df.detected_animal.replace(
        ['american mink'], 'mink')  # norka
    df.detected_animal = df.detected_animal.replace(
        [], 'polecat')  # tchórz
    df.detected_animal = df.detected_animal.replace(
        ['american beaver', 'muskrat', 'north american porcupine',
         'nutria', 'capybara', 'woodchuck'],
        'beaver'
    )
    df.detected_animal = df.detected_animal.replace(
        ['north american river otter', 'common water rat'], 'otter')
    df.detected_animal = df.detected_animal.replace(
        ['american black bear', 'brown bear'], 'bear')
    df.detected_animal = df.detected_animal.replace(
        ['grey fox', 'hoary fox', 'puma', 'red fox'], 'fox')
    df.detected_animal = df.detected_animal.replace(
        'domestic cat', 'cat')
    df.detected_animal = df.detected_animal.replace(
        'domestic dog', 'dog')
    df.detected_animal = df.detected_animal.replace(
        ['wild cat', 'leopard cat'], 'wildcat')
    df.detected_animal = df.detected_animal.replace(
        ['grey wolf', 'coyote', 'golden jackal'], 'wolf')
    df.detected_animal = df.detected_animal.replace(
        ['eurasian badger', 'palawan stink-badger', 'greater hog badger'],
        'badger')
    df.detected_animal = df.detected_animal.replace(
        ['common duiker', 'elk', 'sambar', 'sika deer',
         'sitatunga', 'white-tailed deer'],
        'red deer'
    )
    df.detected_animal = df.detected_animal.replace(
        ['bobcat', 'eurasian lynx'], 'lynx')
    df.detected_animal = df.detected_animal.replace(
        ['bushbuck', 'european roe deer', 'large-antlered muntjac',
         'mule deer', 'pampas deer', 'puku'],
        'roe deer'
    )
    df.detected_animal = df.detected_animal.replace(
        ['northern raccoon'], 'raccoon dog')
    df.detected_animal = df.detected_animal.replace(
        ['american bison', 'european bison', 'african buffalo'], 'bison')
    df.detected_animal = df.detected_animal.replace(
        ['common fallow deer'], 'fallow deer')
    df.detected_animal = df.detected_animal.replace(
        ['cervidae family'], 'moose')

    df.detected_animal = df.detected_animal.replace([
        'striped skunk', 'phillipine porcupine', 'indian crested porcupine',
        'virginia opossum', 'bat', 'domestic horse', 'domestic cattle',
        'blue monkey', 'cetartiodactyla order', 'domestic goat',
        'domestic sheep', 'golden snub-nosed monkey', 'hartebeest',
        'hoary marmot', 'llama', 'mammal', 'masked palm civet', 'mouflon',
        'mountain tapir', 'nine-banded armadillo', 'northern chamois',
        'palawan treeshrew', 'philippine pangolin', 'possum family',
        'pronghorn', 'reeves\' muntjac', 'sable antelope', 'serval',
        'south american coati', 'spotted hyaena', 'western gray kangaroo',
        'white-footed mouse', 'white-tailed mongoose',
        'arizona black-tailed prairie dog', 'ferret badger species',
        'common palm civet'],
        'other'
    )
    df.detected_animal = df.detected_animal.replace('blank', 'empty')

    return df
