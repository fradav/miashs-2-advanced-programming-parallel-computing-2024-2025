#r "nuget: FSharp.Data"
#r "nuget: Fli"
#r "nuget: FSharp.Collections.ParallelSeq"

open System.IO
open FSharp.Data
open Fli
open FSharp.Collections.ParallelSeq

let args = fsi.CommandLineArgs

let tikzPath = args[1]
let sourcePath = Path.Combine(__SOURCE_DIRECTORY__, tikzPath)

Directory.GetFiles(sourcePath)
|> Seq.filter (fun f -> not (f.EndsWith(".tex")))
|> Seq.iter File.Delete


[<Literal>]
let settingsPath = __SOURCE_DIRECTORY__ + "/../.vscode" + "/settings.json"

type LatexTasksT = JsonProvider<settingsPath>

let latexTasks = LatexTasksT.GetSample().LatexWorkshopLatexRecipes[0].Tools

let executeOnFile l o p =
    let replaced =
        l
        |> List.skip 1
        |> List.map (fun (v: string) -> v.Replace("%OUTDIR%", "."))
        |> List.map (fun (v: string) -> v.Replace("%DOC%", p))
        |> List.map (fun (v: string) -> v.Replace("%DOCFILE%", Path.GetFileNameWithoutExtension(p)))
    cli {
        Exec (List.head l)
        Arguments replaced
        WorkingDirectory sourcePath
    }
    |> Command.execute
    |> printfn "%A"

let latexTools = 
    LatexTasksT.GetSample().LatexWorkshopLatexTools
    |> Seq.filter (fun t -> Array.exists (fun l -> l = t.Name) latexTasks)
    |> Seq.map (fun t -> t.Name, t.Command::(Array.toList t.Args) )
    |> Map.ofSeq


Directory.GetFiles(sourcePath)
|> Seq.filter (fun f -> (f.EndsWith(".tex")))
|> PSeq.iter (fun f -> latexTasks |> Seq.iter (fun t -> executeOnFile (Map.find t latexTools) tikzPath f))
