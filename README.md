# KanjiOCR
Skriptsprachen Projekt FH SWF 2019/20. Schriftzeichenerkennung mit Machine Learning in Python

Zur Verwendung sollte die Dokumentation ausreichen. Trotzdem folgt noch eine kleine Anleitung.

Die nötigen Daten können unter http://etlcdb.db.aist.go.jp/ gefunden werden (passwortgeschützt).

Es werden benötigt bzw wurde verwendet: Tensorflow 2.1 mit Python 3.7.6 und scitkit-learn 0.22.1.

Um die benötigte Zeit für das Training von Modellen zu verkürzen, wird empfohlen den GPU Support zu verwenden. Eine Anleitung ist unter https://www.tensorflow.org/install/gpu zu finden.

Die DataConverter Klasse dient nur dazu, die Datenbanken (derzeit nur ETL1) in ein anderes Format zu bringen und in den Speicher zu laden. Dies soll es vereinfachen, Klassen zu selektieren. Desweiteren trennt die exportPNGOrganised Methode Daten zufällig über eine Methode von sklearn in Trainings- und Testdaten.
Ein einmaliger Aufruf von DataConverter.split() ist erforderlich, danach kann die DataConverter.load() Methode aufgerufen werden, um alle Klassen mit ihren Features getrennt in den Speicher zu laden. Um das derzeitige Modell zu verwenden, welches in der Klasse CNN_ETL1 zu finden ist, muss einmalig die DataConverter.exportPNGOrganised() Methode aufgerufen werden.
Ein Modell eines CNN_ETL1 Objektes kann über die train() trainiert werden und man kann Modelle speichern und laden.
