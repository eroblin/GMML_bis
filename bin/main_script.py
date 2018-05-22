import os
import click
import yaml
import logging
import logging.config
from scripts.build_wordnet_data import get_word_dependencies, generate_network
from scripts.Poincare import Poincare_Embeddings


def setup_logging(default_path='logging.yaml', default_level=logging.INFO, env_key='LOG_CFG'):
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


@click.group()
def cli():
    pass


@click.command(name='fetch-wordnet-data')
@click.option('--word', default='music_genre')
def fetch_wordnet_data(word):
    get_word_dependencies(word)
    generate_network(word)


@click.command(name='generate-poincare-embeddings')
@click.option('--dataset', default='music_genre')
@click.option('--nb_epochs', default=100)
@click.option('--nb_neg_samplings', default=10)
@click.option('--learning_rate', default=.1)
def generate_poincare_embeddings(dataset, nb_epochs, nb_neg_samplings, learning_rate):
    Poincare_Embeddings(dataset, nb_epochs, nb_neg_samplings, learning_rate)


cli.add_command(fetch_wordnet_data)
cli.add_command(generate_poincare_embeddings)

if __name__ == '__main__':
    setup_logging()
    cli()
