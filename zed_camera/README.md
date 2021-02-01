# Phidippides
Documentatie ZED-code
Deze documentatie is ook te vinden in de Teams onder 'Team_Phidippides->Quadriga->2020-2021->Technische Informatica->project56'

Inleiding

Tijdens het project is er software geschreven voor de ZED-camera. Deze software draait het objectdetectie model en bepaald de afstand tot gedetecteerde objecten. Dit is samengevoegd tot één programma. In dit document worden de verschillende onderdelen van de code toegelicht. 

Functionaliteiten

De geschreven software draait het objectdetectie model en bepaald de afstand tot de gedetecteerde objecten. Het programma kan werken met ieder tensorflow model. In de code staat deze nu gekoppeld aan ons eigen getrainde model, maar dat kan gemakkelijk veranderd worden als er een ander model gebruikt moet worden. 

ZED-camera Python API

Voor de software wordt gebruik gemaakt van de ZED-camera Python API. Met deze API kan de camera in Python bestuurd worden. De API wordt in de geschreven software gebruikt om beelden en diepte informatie van de camera op te vragen. Om de camera te gebruiken moet er eerst wat initialisatie gedaan worden:
Met ‘init = sl.InitParameters()’ worden de standaard parameters ingesteld.
Met ‘init.coordinate_units = sl.UNIT.METER’ wordt de afstand gemeten in meters. Standaard is dit in millimeters.
Met image = sl.Mat() wordt een matrix gedeclareerd waar de dieptekaart of het beeld van de camera in opgeslagen kan worden. sl.Mat() is een datatype van de ZED API. Deze kan alleen voor de ZED-camera gebruikt worden.

Na de initialisatie kan er een frame of dieptekaart worden opgevraagd van de camera:
Met ‘cam.retrieve_image(image, sl.VIEW.RIGHT)’ wordt een frame van de rechter camera opgevraagd.
Met ‘cam.retrieve_measure(depthMap, sl.MEASURE.DEPTH)’ wordt de dieptekaart van de camera opgevraagd. (Standaard van de rechter camera)

Het eerste argument dat aan de bovenste twee functies wordt meegegeven is een sl.Mat(). Hier wordt de opgevraagde frame of dieptekaart in opgeslagen.
Object detectie
In de Git staan twee versies van de code: een die met ons eigen model werkt en een die een model van internet kan downloaden. Het model van internet downloaden is handig om de code te testen. Om een ander model te downloaden moeten de MODEL_DOWNLOAD_BASE, MODEL_NAME en MODEL_DATE aangepast worden. Er zijn verschillende modellen te vinden op de TensorFlow Object Detection Model Zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). 
Als je een ander lokaal model wil gebruiken, moet de andere code gebruikt worden. Hier moeten dan de PATH_TO_CKPT, PATH_TO_CFG en PATH_TO_LABELS aangepast worden. Deze moeten dan verwijzen naar respectievelijk de checkpoint van het model, de pipeline configuratie en de labelmap. 

De object detectie wordt aangeroepen door:
input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
detections, predictions_dict, shapes = detect_fn(input_tensor)

image_np is de frame die van de ZED-camera afkomstig is. Omdat er niet direct een sl.Mat() als invoer voor de TensorFlow functie gebruikt kan worden, wordt het plaatje eerst omgezet naar een numpy array. Dit gebeurt door de ‘load_image_into_numpy_array’ functie. 

Omdat we de afstand tot de gedetecteerde objecten willen bepalen, hebben wij de coördinaten nodig van de boxes rond de objecten. Deze worden opgeslagen in de array ‘getDepthFromBox(detections['detection_boxes'][0].numpy()’. Dit is dus een array met alle coördinaten van de gedetecteerde objecten. De coördinaten bestaan uit 4 waardes: xMin, xMax, yMin, yMax. Dit zijn de coördinaten van de hoekpunten van de boxes.
Dieptekaart
De dieptekaart van de ZED-camera is een matrix met diepte informatie van iedere pixel van de camera. Standaard wordt de dieptekaart van de rechter camera opgevraagd.  De dieptekaart is een matrix met floats. De diepte van een bepaalde pixel met de coördinaten (x,y) kan opgevraagd worden door ‘depthMap.get_value(y, x)’ aan te roepen. 

Met de coördinaten die uit het object detectie model komen kan de afstand tot een object bepaald worden. Dit gebeurt in de functie ‘getDepthFromBox()’. Op dit moment wordt alleen het midden van een box bepaald, en dan de afstand opgevraagd. Er kan ook een ingewikkelder algoritme toegevoegd worden. Door bijvoorbeeld het gemiddelde te berekenen, of waardes die afwijken van de waardes van de andere pixels in de box niet mee te nemen. Dit zijn verbeteringen die in de toekomst toegevoegd kunnen worden.
