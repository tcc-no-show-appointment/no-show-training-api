import sys
import os
import pytest
import pandas as pd
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.services.data_validation import DataValidator


MINIMAL_ROW = {
    'id': 1,
    'Status': 'Realizado',
    'Marcacao': '2024-11-16T08:00:00',
    'DataHoraConsulta': '2024-11-23T14:00:00',
    'Idade': 62,
    'Sexo': 'F',
    'CidadePaciente': 'SAO PAULO',
    'BairroPaciente': 'BELA VISTA',
    'TipoConvenio': 'Enfermaria',
    'idUnicoPaciente': 'ID001',
    'UnidadeAtendimento': 'CAMPO BELO',
    'EnderecoUnidadeAtendimento': 'RUA VIEIRA DE MORAES',
    'CEPUnidadeAtendimento': '04617-015',
    'Especialidade': 'CARDIOLOGIA',
}


class TestDataValidator:
    def _csv_with(self, row: dict) -> str:
        df = pd.DataFrame([row])
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(tmp.name, index=False)
        tmp.close()
        return tmp.name

    def test_valid_csv_passes(self):
        path = self._csv_with(MINIMAL_ROW)
        try:
            v = DataValidator()
            result, df = v.load_and_validate(path)
            assert result['is_valid'] is True
            assert df is not None
            assert len(df) == 1
        finally:
            os.unlink(path)

    def test_missing_required_column_fails(self):
        row = {k: v for k, v in MINIMAL_ROW.items() if k != 'Especialidade'}
        path = self._csv_with(row)
        try:
            v = DataValidator()
            result, _ = v.load_and_validate(path)
            assert result['is_valid'] is False
            assert any('Especialidade' in e for e in result.get('missing_columns', []) + result.get('errors', []))
        finally:
            os.unlink(path)

    def test_nonexistent_file_fails(self):
        v = DataValidator()
        result = v.validate_file('/tmp/nonexistent_file_abc123.csv')
        assert result['is_valid'] is False
        assert result['errors']

    def test_unsupported_format_fails(self):
        tmp = tempfile.NamedTemporaryFile(suffix='.txt', delete=False)
        tmp.write(b'col1,col2\n1,2\n')
        tmp.close()
        try:
            v = DataValidator()
            result = v.validate_file(tmp.name)
            assert result['is_valid'] is False
        finally:
            os.unlink(tmp.name)

    def test_multiple_rows(self):
        rows = [MINIMAL_ROW.copy() for _ in range(5)]
        for i, r in enumerate(rows):
            r['id'] = i + 1
        path = self._csv_with(rows[0])  # write first row only via helper
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        try:
            v = DataValidator()
            result, loaded = v.load_and_validate(path)
            assert result['is_valid'] is True
            assert len(loaded) == 5
        finally:
            os.unlink(path)
