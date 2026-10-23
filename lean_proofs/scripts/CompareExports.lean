import Main

open Lean Elab Command in
run_cmd do
  let some configPath ← IO.getEnv "EF21MUON_COMPARATOR_CONFIG"
    | throwError "EF21MUON_COMPARATOR_CONFIG is not set"
  let some challengePath ← IO.getEnv "EF21MUON_COMPARATOR_CHALLENGE"
    | throwError "EF21MUON_COMPARATOR_CHALLENGE is not set"
  let some solutionPath ← IO.getEnv "EF21MUON_COMPARATOR_SOLUTION"
    | throwError "EF21MUON_COMPARATOR_SOLUTION is not set"
  let configText ← IO.FS.readFile configPath
  let config : Comparator.Config ← IO.ofExcept <|
    Lean.FromJson.fromJson? (← IO.ofExcept (Lean.Json.parse configText))
  let challenge ← IO.FS.readFile challengePath
  let solution ← IO.FS.readFile solutionPath
  let cwd ← IO.Process.getCurrentDir
  let context : Comparator.Context := {
    projectDir := cwd
    challengeModule := config.challenge_module.toName
    solutionModule := config.solution_module.toName
    theoremNames := config.theorem_names.map String.toName
    definitionNames := #[]
    legalAxioms := config.permitted_axioms.map String.toName
    leanPrefix := cwd
    gitLocation := cwd
    whichLandrun := ""
    whichLean4Export := ""
    externalKernels := {} }
  let action : IO Unit := ReaderT.run (Comparator.verifyMatch challenge solution) context
  liftIO action
  IO.println "Comparator accepted the frozen statements and replayed their proofs."
