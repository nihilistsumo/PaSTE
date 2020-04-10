import requests
import argparse

def get_category(page, session):
    URL = "https://en.wikipedia.org/w/api.php"
    trash_cats = ["All Wikipedia", "Articles", "articles", "Archives", "archives", "CS1", "Commons category", "Use dmy",
                  "Use mdy", "Coordinates", "Redirects", "Webarchive", "All pages", "Pages"]
    PARAMS = {
    "action": "query",
    "format": "json",
    "prop": "categories",
    "titles": page
    }
    cats = []

    R = session.get(url=URL, params=PARAMS)
    DATA = R.json()

    for v in DATA["query"]["pages"].values():
        if 'categories' in v.keys():
            for c in v['categories']:
                catname = c['title'].split(':')[1]
                if not any(t in catname for t in trash_cats):
                    cats.append(c['title'].split(':')[1])
    return cats

def convert(qry_attn_file, out_file):
    f = True
    s = requests.Session()
    lines = []
    c= 0
    with open(qry_attn_file, 'r') as tr:
        for l in tr:
            page = l.split('\t')[1]
            cat_list = get_category(page, s)
            lines.append(l.split('\t')[0]+'\t'+l.split('\t')[1]+'. '+'. '.join(cat_list)+'\t'+l.split('\t')[2]+'\t'+l.split('\t')[3])
            c += 1
            if c%100 == 0:
                print(c)
    with open(out_file, 'w') as out:
        for l in lines:
            out.write(l)

def main():
    parser = argparse.ArgumentParser(
        description='Convert qry attn data to qry attn with category info')
    parser.add_argument('-qf', '--qry_attn_file', help='Path to input query attn file')
    parser.add_argument('-out', '--outfile', help='Path to converted output file')
    args = vars(parser.parse_args())
    qry_attn_file = args['qry_attn_file']
    outfile = args['outfile']
    convert(qry_attn_file, outfile)

if __name__ == '__main__':
    main()