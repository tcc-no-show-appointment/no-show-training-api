REQUIRED_COLUMNS = [
    "id",
    "Status",
    "Marcacao",
    "DataHoraConsulta",
    "Idade",
    "Sexo",
    "CidadePaciente",
    "BairroPaciente",
    "TipoConvenio",
    "idUnicoPaciente",
    "UnidadeAtendimento",
    "EnderecoUnidadeAtendimento",
    "CEPUnidadeAtendimento",
    "Especialidade"
]

SUPPORTED_FORMATS = {
    "csv": "text/csv",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "xls": "application/vnd.ms-excel",
    "parquet": "application/octet-stream"
}
