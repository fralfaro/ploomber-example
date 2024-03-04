import app


def test_functions_called_at_least_once(mocker):
    # Simulates the specified functions to track calls
    mocker.patch("app.cs_sidebar")
    mocker.patch("app.cs_body")

    # Calls the main function
    app.main()

    # Check if 'cs_sidebar' and 'cs_body' were called at least once
    assert app.cs_sidebar.called
    assert app.cs_body.called
