User-visible changes
--------------------

V1.1.0:

- [#3]  Accept 'delete 2-5' to delete a range of jobs
- [#25] Replace 'inline' command (resp. attribute) in the Interactive
        (resp. library) with 'inputs'.  Deprecate 'inline'.
- [#4]  Introduce 'outputs' command to change the type of
        outputs from inline to data-assets.
- [#21] Read credentials from file $DOWML_CREDENTIALS_FILE
        as last resort.


V1.0.0:

- Packaging-only changes


V0.9.0, first release on PyPi:

- [#17] DOWMLLib now returns tabular outputs as dataframes by default.
        Also replace the now-deprecated csv_as_dataframe with tabular_as_csv
- [#18] DOWMLLib.get_output returns a dict instead of a list
- [#16] 'output' stores files in subdirectories
- [#12] Add 'shell' command in the Interactive