#!/usr/bin/env python
# coding: utf-8
from bert import Bert

#get the context from the wikipedia page: https://en.wikipedia.org/wiki/Eindhoven
context = "Eindhoven (/ˈaɪnthoʊvən/ EYENT-hoh-vən;[8] Dutch: [ˈɛintˌɦoːvə(n)] ⓘ) is a city and municipality of the Netherlands, located in the southern province of North Brabant, of which it is the largest municipality, and is also located in the Dutch part of the natural region the Campine. With a population of 249,054 (1 January 2025) on a territory of 88.92 km2,[9] it is the fifth-largest city of the Netherlands and the largest outside the Randstad conurbation.Eindhoven was originally located at the confluence of the Dommel and the Gender.[10][11][12] A municipality since the 13th century, Eindhoven witnessed rapid growth starting in the 1900s by textile and tobacco industries. Two well-known companies, DAF Trucks and Philips, were founded in the city; Philips would go on to become a major multinational conglomerate while based in Eindhoven.[13] Apart from Philips, Eindhoven also contains the globally famous Design Academy Eindhoven.[14] Neighbouring cities and towns include Son en Breugel, Nuenen, Geldrop-Mierlo, Helmond, Heeze-Leende, Waalre, Veldhoven, Eersel, Oirschot and Best. The agglomeration has a population of 337,487. The metropolitan area consists of 780,611 inhabitants. The city region has a population of 753,426. The Brabantse Stedenrij combined metropolitan area has about two million inhabitants. Etymology. The name may derive[15] from the contraction of the regional words eind (meaning \"last\" or \"end\") and hove (or hoeve, a section of some 14 hectares of land). "


# ask bert
bert = Bert()

question = "what is the population of Eindhoven?"
predicted_answer = bert.answer_me(context, question)
print(f"Question '{question}'")
print(f"Predicted Answer: '{predicted_answer}'")
print(f"Correct answer '249,054'")

question = "where was Eindhoven originally located?"
predicted_answer = bert.answer_me(context, question)
print(f"Question: '{question}'")
print(f"Predicted Answer: '{predicted_answer}'")
print(f"Correct answer 'at the confluence of the Dommel and the Gender'")
