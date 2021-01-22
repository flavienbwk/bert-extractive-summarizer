from summarizer import Summarizer
from summarizer.coreference_handler import CoreferenceHandler
import argparse


sentence_handlers = {
    "en": "en_core_web_sm",
    "fr": "fr_core_news_sm"
}


def run():
    parser = argparse.ArgumentParser(description='Process and summarize lectures')
    parser.add_argument('-path', dest='path', default=None, help='File path of lecture')
    parser.add_argument('-lang', dest='lang', default='en', help='Language supported : en,fr')
    parser.add_argument('-model', dest='model', default='bert-large-uncased', help='')
    parser.add_argument('-num-sentences', dest='num_sentences', default=-1, help='Will return X sentences')
    parser.add_argument('-ratio', dest='ratio', default=-1, help='Will return a ratio of sentences from the text length (0.2 is a good value)')
    parser.add_argument('-hidden', dest='hidden', default=-2, help='Which hidden layer to use from Bert')
    parser.add_argument('-reduce-option', dest='reduce_option', default='mean', help='How to reduce the hidden layer from bert')
    parser.add_argument('-greedyness', dest='greedyness', help='Greedyness of the NeuralCoref model', default=0.45)
    args = parser.parse_args()

    if not args.path:
        raise RuntimeError("Must supply text path.")

    with open(args.path) as d:
        text_data = d.read()

    spacy_model = sentence_handlers[args.lang]
    model = Summarizer(
        model=args.model,
        hidden=args.hidden,
        reduce_option=args.reduce_option,
        sentence_handler=CoreferenceHandler(spacy_model=spacy_model)
    )
    
    if int(args.num_sentences) >= 0:
        result = model(text_data, num_sentences=int(args.num_sentences))
    elif float(args.ratio) >= 0.0:
        result = model(text_data, ratio=float(args.num_sentences))
    else:
        result = model(text_data)
    print(result)


if __name__ == '__main__':
    run()

