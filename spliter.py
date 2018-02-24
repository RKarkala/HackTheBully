import xml.etree.cElementTree
import html

with open('bully.tsv', 'w', encoding='utf-8') as f:
    f.write("Message\tRating\n")
    file = xml.etree.cElementTree.parse("bully.xml").getroot()
    for post in file.iter('POST'):
        normal_text = html.unescape(post.find('TEXT').text)
        normal_text = normal_text.replace("<br>", "")
        normal_text = normal_text.replace("Q:", "")
        normal_text = normal_text.replace("A:", "")
        average_rating = 0.0
        for rating in post.iter('SEVERITY'):
            try:
                average_rating += float(rating.text)
            except:
                average_rating += 0.0
        average_rating /= 3
        if average_rating >= 2:
            average_rating = 2
        elif average_rating > 0:
            average_rating = 1
        else:
            average_rating = 0
        print(normal_text, average_rating)
        f.write(normal_text+"\t"+str(average_rating)+"\n")

