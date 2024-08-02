# Classificazione manuale delle tecniche

## Introduzione

La seguente è una guida per chiunque sia interessato ad aiutare ad espandere il dataset delle tecniche usato dal modello.

### Classificazione

La classificazione è il processo di assegnare una o più etichette a un insieme di dati.

Implicitamente, tutti gli atleti sono capaci di classificare:
vedendo solamente un'azione, un atleta è in grado di riconoscere un eventuale tecnica eseguita e associarla a un nome.

Nel caso di questo progetto, la classificazione è il processo di assegnare un nome a una tecnica eseguita in un video.

### Dataset

Il dataset, in questo caso l'insieme dei dati contenente esempi di tecniche,
è la base da cui un modello di intelligenza artificiale impara a riconoscere ciò che gli viene mostrato.

- Il nostro dataset inizia dai video delle competizioni dell'[EJU],
da cui volontari estraggono dei segmenti di video in cui vengono eseguite le tecniche,
assegnandogli inoltre informazioni aggiuntive come il nome della tecnica e l'atleta (bianco o blu) che la esegue.

- In seguito, un modulo del modello analizza i segmenti di video per estrarre dati significativi,
ad esempio la posizione dei vari arti, che verranno poi usati per addestrare effettivamente il modello.

## Classificazione manuale

La prima fase del processo di creazione del dataset necessita di volontari
che manualmente selezionino e classifichino i segmenti in cui vengono eseguite delle tecniche.
A tale scopo è stato creato un modulo che permette,
a partire dal video completo di una competizione e dall'elenco di tecniche eseguite nella competizione,
di dividere automaticamente il video nei segmenti in cui vengono eseguite le tecniche.

Il compito dei volontari è quello di identificare i segmenti ed elencarli in dei file di testo
con un formato speciale, in modo che il modulo possa elaborarli successivamente.

### Formato file

Il file in cui devono essere elencate le tecniche necessita di seguire un formato specifico, detto `csv`.
Brevemente, tale formato può essere elaborato come un foglio di calcolo tramite un programma come Excel.
È possibile scaricare [questo file][CSV_TEMPLATE] di esempio da cui iniziare
([video competizione][SAR2210D2T1FIN]).

<img src="/doc/img/excel_template.png" style="display: block; margin: auto;"/>

Le colonne del file sono le seguenti:

- `competition`: il codice assegnato alla competizione
- `ts_start`: il tempo di inizio della tecnica (formato `MM:SS` o `HH:MM:SS`)
- `ts_end`: il tempo di fine della tecnica (formato `MM:SS` o `HH:MM:SS`)
- `throw`: il nome della tecnica eseguita
- `tori`: il colore dell'atleta che esegue la tecnica (`white`/`blue`)

> **Attenzione** durante l'uso di programmi come Excel:
> le colonne temporali (`ts_start` e `ts_end`) potrebbero essere interpretate come date
> e quindi formattate erroneamente. Per evitare ciò,
> si consiglia di impostare il formato delle colonne come `testo`.
> 
> <img src="/doc/img/excel_text_column.png" style="display: block; margin: auto;"/>

## Esempi

I seguenti video sono estratti dalle finali della competizione
[_European Judo Championships U23 Sarajevo 2022_][SAR2210D2T1FIN]
dell'[EJU], Giorno 2 - Tatami 1,
e riportano le classificazioni indicate nel [file di esempio][CSV_TEMPLATE] scaricabile.

https://github.com/user-attachments/assets/aa82cf9c-1bcd-4751-8044-564e5f17bbf0

https://github.com/user-attachments/assets/4f06e968-541d-45dc-89da-ff9e127a1355

https://github.com/user-attachments/assets/195e86c3-e0c9-41ef-9783-082fe3a205ca

https://github.com/user-attachments/assets/26f10761-b9d8-4fad-bf9e-975f868f4b76


[CSV_TEMPLATE]: /doc/template.csv
[EJU]: https://www.eju.net/
[SAR2210D2T1FIN]: https://www.youtube.com/watch?v=ozvTsftwfGg
