# test_app.py
import app

def test_functions_called_at_least_once(mocker):
    # Simula las funciones especificadas para rastrear las llamadas
    mocker.patch('app.cs_sidebar')
    mocker.patch('app.cs_body')

    # Llama a la función principal
    app.main()

    # Verifica si cs_sidebar y cs_body fueron llamadas al menos una vez
    assert app.cs_sidebar.called
    assert app.cs_body.called
