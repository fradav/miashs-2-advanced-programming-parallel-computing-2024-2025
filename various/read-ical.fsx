#r "nuget: Ical.Net, 4.2.0"

// Deserialize "Calcul Parallèle.ics" file
// Print date/time of each events

open Ical.Net
open System.IO

let printEvent (ev: CalendarComponents.CalendarEvent) =
    let day = ev.DtStart.Date.ToString("dd/MM/yyyy")
    let hstart = ev.DtStart.Value.ToString("HH\\hmm")
    let hend = ev.DtEnd.Value.ToString("HH\\hmm")
    $"{day} {hstart}-{hend}"

let printEvents (path: string) =
    let icaltext = File.ReadAllText(path)
    let file = Path.GetFileNameWithoutExtension(path)
    let events = Calendar.Load(icaltext)
    events.Events
    |> Seq.sortBy _.DtStart.Date.ToFileTimeUtc()
    |> Seq.map printEvent
    |> (fun x -> File.WriteAllLines($"various/{file}.txt", x))

printEvents @"various\Calcul Parallèle.ics"
printEvents @"various\Calcul Parallèle-2.ics"

let printEventDuration (ev:CalendarComponents.CalendarEvent) =
    let day = ev.DtStart.Value.ToString("dd/MM/yyyy dddd")
    let duration = ev.Duration.TotalHours |> sprintf "%0.1f"
    day + " " + duration

