# -*- coding: utf-8 -*-
"""
LibreOffice Macro - Portfolio Optimization Client
Communicates with remote server for computation
"""

import uno
import json
import urllib.request
import urllib.parse
from math import comb

# Server configuration
SERVER_URL = "https://your-app-name.onrender.com"  # Replace with your Render.com URL

def leggi_dati_da_foglio_calc(ctx=None):
    """Read data from LibreOffice Calc sheet"""
    import uno
    from math import comb

    # Get LibreOffice context
    ctx = uno.getComponentContext()
    smgr = ctx.ServiceManager
    desktop = smgr.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
    model = desktop.getCurrentComponent()
    sheets = model.Sheets
    sheet = model.CurrentController.ActiveSheet

    # Read capital and capital units
    capitale = int(sheet.getCellRangeByName("B1").Value)
    unita_capitale = int(sheet.getCellRangeByName("B2").Value)

    # Read options from row 3 (index 2)
    opzioni = []
    coefficienti = []
    start_row = 2
    i = 0
    while True:
        row = start_row + i
        nome = sheet.getCellByPosition(3, row).String  # Column D
        if not nome:
            break

        # Check that a, b, c are also present
        try:
            a = float(sheet.getCellByPosition(4, row).Value)  # Column E
            b = float(sheet.getCellByPosition(5, row).Value)  # Column F
            c = float(sheet.getCellByPosition(6, row).Value)  # Column G
        except Exception:
            raise ValueError(f"Error: missing coefficient in row {row + 1} for option '{nome}'.")

        opzioni.append({"name": nome, "a": a, "b": b, "c": c})
        coefficienti.append((a, b, c))
        i += 1

    numero_opzioni = len(opzioni)
    sheet.getCellRangeByName("B3").Value = numero_opzioni

    # Read antisynergies starting from row 3 (index 2)
    antisinergie = []
    row = 2
    while True:
        nome1 = sheet.getCellByPosition(9, row).String  # Column J
        nome2 = sheet.getCellByPosition(10, row).String  # Column K
        if not nome1 and not nome2:
            break
        if not nome1 or not nome2:
            raise ValueError(f"Error: incomplete antisynergy in row {row + 1}.")

        try:
            valore = float(sheet.getCellByPosition(11, row).Value)  # Column L
        except Exception:
            raise ValueError(f"Error: missing antisynergy value in row {row + 1}.")
        
        antisinergie.append({"option1": nome1, "option2": nome2, "value": valore})
        row += 1

    # Calculate and write combinations
    combinazioni = comb(unita_capitale + numero_opzioni - 1, numero_opzioni - 1)
    sheet.getCellRangeByName("B5").Value = combinazioni

    # Write to Test sheet
    nome_test = "Test"
    try:
        test_sheet = sheets.getByName(nome_test)
    except:
        sheets.insertNewByName(nome_test, len(sheets))
        test_sheet = sheets.getByName(nome_test)
    
    test_sheet.getCellRangeByName("A1").String = "Capitale"
    test_sheet.getCellRangeByName("B1").Value = capitale
    test_sheet.getCellRangeByName("A2").String = "Unità di Capitale"
    test_sheet.getCellRangeByName("B2").Value = unita_capitale
    test_sheet.getCellRangeByName("A3").String = "Numero Opzioni"
    test_sheet.getCellRangeByName("B3").Value = numero_opzioni
    test_sheet.getCellRangeByName("A4").String = "Combinazioni"
    test_sheet.getCellRangeByName("B4").Value = combinazioni

    test_sheet.getCellRangeByName("A6").String = "Nome Opzione"
    test_sheet.getCellRangeByName("B6").String = "a"
    test_sheet.getCellRangeByName("C6").String = "b"
    test_sheet.getCellRangeByName("D6").String = "c"

    for i, opzione in enumerate(opzioni):
        test_sheet.getCellByPosition(0, 6 + i).String = opzione["name"]
        test_sheet.getCellByPosition(1, 6 + i).Value = opzione["a"]
        test_sheet.getCellByPosition(2, 6 + i).Value = opzione["b"]
        test_sheet.getCellByPosition(3, 6 + i).Value = opzione["c"]

    if antisinergie:
        test_sheet.getCellRangeByName("F6").String = "Antisinergia 1"
        test_sheet.getCellRangeByName("G6").String = "Antisinergia 2"
        test_sheet.getCellRangeByName("H6").String = "Valore"
        for j, anti in enumerate(antisinergie):
            test_sheet.getCellByPosition(5, 6 + j).String = anti["option1"]
            test_sheet.getCellByPosition(6, 6 + j).String = anti["option2"]
            test_sheet.getCellByPosition(7, 6 + j).Value = anti["value"]

    return capitale, unita_capitale, numero_opzioni, opzioni, antisinergie, combinazioni

def call_optimization_server(capitale, unita_capitale, opzioni, antisinergie, penalty_strength=1.0):
    """Call the remote optimization server"""
    
    # Prepare request data
    request_data = {
        "capitale": capitale,
        "unita_capitale": unita_capitale,
        "opzioni": opzioni,
        "antisinergie": antisinergie,
        "penalty_strength": penalty_strength
    }
    
    # Convert to JSON
    json_data = json.dumps(request_data).encode('utf-8')
    
    try:
        # Make HTTP request
        req = urllib.request.Request(
            f"{SERVER_URL}/optimize",
            data=json_data,
            headers={
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            method='POST'
        )
        
        with urllib.request.urlopen(req, timeout=300) as response:  # 5 minute timeout
            if response.getcode() == 200:
                result = json.loads(response.read().decode('utf-8'))
                return result
            else:
                raise Exception(f"Server returned status code: {response.getcode()}")
                
    except urllib.error.URLError as e:
        raise Exception(f"Network error: {e}")
    except json.JSONDecodeError as e:
        raise Exception(f"Invalid JSON response: {e}")
    except Exception as e:
        raise Exception(f"Server communication error: {e}")

def scrivi_qubo_in_foglio_calc(qubo_matrix, variables, offset, nome_foglio="QUBO_Matrix"):
    """Write QUBO matrix to LibreOffice Calc sheet"""
    import uno

    # Get LibreOffice context
    ctx = uno.getComponentContext()
    smgr = ctx.ServiceManager
    desktop = smgr.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
    model = desktop.getCurrentComponent()
    sheets = model.Sheets

    # Create new sheet
    try:
        ws = sheets.getByName(nome_foglio)
    except:
        sheets.insertNewByName(nome_foglio, len(sheets))
        ws = sheets.getByName(nome_foglio)

    # Write offset
    ws.getCellByPosition(0, 0).String = "Offset"
    ws.getCellByPosition(1, 0).Value = float(offset)

    n = len(variables)

    # Write column headers
    for j, var in enumerate(variables):
        ws.getCellByPosition(j+1, 1).String = str(var)

    # Write row headers and matrix
    for i, var in enumerate(variables):
        ws.getCellByPosition(0, i+2).String = str(var)
        for j in range(n):
            ws.getCellByPosition(j+1, i+2).Value = float(qubo_matrix[i][j])

    print(f"✅ QUBO matrix written to sheet '{nome_foglio}'")

def write_results_to_calc(server_response, opzioni_names):
    """Write optimization results to LibreOffice Calc"""
    import uno

    # Connect to Calc
    ctx = uno.getComponentContext()
    smgr = ctx.ServiceManager
    desktop = smgr.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
    model = desktop.getCurrentComponent()
    sheets = model.Sheets

    # Results sheet
    results_sheet_name = "Risultati"
    if sheets.hasByName(results_sheet_name):
        sheets.removeByName(results_sheet_name)
    sheets.insertNewByName(results_sheet_name, len(sheets))
    results_sheet = sheets.getByName(results_sheet_name)

    # Headers
    results_sheet.getCellRangeByName("A1").String = "Method"
    results_sheet.getCellRangeByName("B1").String = "Objective"
    for j, nome_opzione in enumerate(opzioni_names):
        results_sheet.getCellByPosition(2 + j, 0).String = f"{nome_opzione} (fractions)"

    # Write all results
    for i, result in enumerate(server_response["results"]):
        results_sheet.getCellByPosition(0, i + 1).String = result["method_name"]
        results_sheet.getCellByPosition(1, i + 1).Value = result["energy"]
        
        # Decode allocations from the sample
        allocations = [0] * len(opzioni_names)
        for key, value in result["sample"].items():
            if value and key.startswith("('x_") and "'," in key:
                # Parse keys like "('x_0', 1)" or "('x_0', 2)"
                try:
                    parts = key.replace("(", "").replace(")", "").replace("'", "").split(", ")
                    if len(parts) >= 2:
                        var_part = parts[0]  # x_0
                        weight_part = parts[1]  # 1, 2, 4, etc.
                        if var_part.startswith("x_"):
                            var_idx = int(var_part.split("_")[1])
                            weight = int(weight_part)
                            if var_idx < len(allocations):
                                allocations[var_idx] += weight
                except:
                    continue
        
        for j, alloc in enumerate(allocations):
            results_sheet.getCellByPosition(2 + j, i + 1).Value = alloc

    # Best solution sheet
    best_sheet_name = "Best_Solution"
    if sheets.hasByName(best_sheet_name):
        sheets.removeByName(best_sheet_name)
    sheets.insertNewByName(best_sheet_name, len(sheets))
    best_sheet = sheets.getByName(best_sheet_name)

    # Best solution headers
    best_sheet.getCellRangeByName("A1").String = "Method"
    best_sheet.getCellRangeByName("B1").String = "Objective"
    best_sheet.getCellRangeByName("C1").String = "Total Capital"
    for j, nome_opzione in enumerate(opzioni_names):
        best_sheet.getCellByPosition(3 + j, 0).String = f"{nome_opzione} (fractions)"
    for j, nome_opzione in enumerate(opzioni_names):
        best_sheet.getCellByPosition(3 + len(opzioni_names) + j, 0).String = f"{nome_opzione} (capital)"
    for j, nome_opzione in enumerate(opzioni_names):
        best_sheet.getCellByPosition(3 + 2*len(opzioni_names) + j, 0).String = f"{nome_opzione} (%)"

    # Best solution data
    best_solution = server_response["best_solution"]
    best_sheet.getCellByPosition(0, 1).String = best_solution["method"]
    best_sheet.getCellByPosition(1, 1).Value = best_solution["objective"]
    best_sheet.getCellByPosition(2, 1).Value = best_solution["total_capital"]

    allocations = best_solution["allocations"]
    capital_values = best_solution["capital_values"]
    total_capital = best_solution["total_capital"]

    for j, alloc in enumerate(allocations):
        best_sheet.getCellByPosition(3 + j, 1).Value = alloc
    
    for j, cap_val in enumerate(capital_values):
        best_sheet.getCellByPosition(3 + len(opzioni_names) + j, 1).Value = cap_val
        perc = (cap_val / total_capital * 100) if total_capital > 0 else 0
        best_sheet.getCellByPosition(3 + 2*len(opzioni_names) + j, 1).Value = perc

    # Create charts (simplified version)
    try:
        create_charts(best_sheet, len(opzioni_names), model)
    except Exception as e:
        print(f"Warning: Could not create charts: {e}")

def create_charts(best_sheet, num_options, model):
    """Create charts in the best solution sheet"""
    charts = best_sheet.Charts
    rect = uno.createUnoStruct("com.sun.star.awt.Rectangle")

    def get_chart_by_name(charts, name):
        for n in charts.ElementNames:
            if n == name:
                return charts.getByName(n).EmbeddedObject
        raise RuntimeError(f"Chart {name} not found")

    try:
        # Bar chart for absolute capital
        rect.X, rect.Y, rect.Width, rect.Height = 1000, 1000, 10000, 6000
        cell_range = best_sheet.getCellRangeByPosition(
            3 + num_options, 0, 2 + 2*num_options, 1
        )
        charts.addNewByName("CapitaleBarChart", rect, (cell_range.RangeAddress,), True, True)
        model.calculateAll()

        bar_chart = get_chart_by_name(charts, "CapitaleBarChart")
        bar_diagram = bar_chart.getFirstDiagram()
        series_bar = bar_diagram.getDataSeries()[0]
        series_bar.DataLabel.ShowNumber = True
        series_bar.DataLabel.NumberFormat = 4

        # Pie chart for percentages
        rect.X, rect.Y, rect.Width, rect.Height = 12000, 1000, 10000, 6000
        cell_range = best_sheet.getCellRangeByPosition(
            3 + 2*num_options, 0, 2 + 3*num_options, 1
        )
        charts.addNewByName("CapitalePieChart", rect, (cell_range.RangeAddress,), True, True)
        model.calculateAll()

        pie_chart = get_chart_by_name(charts, "CapitalePieChart")
        pie_diagram = pie_chart.getFirstDiagram()
        series_pie = pie_diagram.getDataSeries()[0]
        series_pie.DataLabel.ShowNumber = False
        series_pie.DataLabel.ShowPercent = True
        series_pie.DataLabel.ShowCategory = True

    except Exception as e:
        print(f"Chart creation error: {e}")

def esegui_qubo_macro_server(*args):
    """
    Main macro function that uses the remote server for optimization
    """
    try:
        # Read data from LibreOffice Calc
        print("Reading data from Calc...")
        (capitale, unita_capitale, numero_opzioni, opzioni,
         antisinergie, combinazioni) = leggi_dati_da_foglio_calc()

        opzioni_names = [opt["name"] for opt in opzioni]
        
        print(f"Data loaded: {numero_opzioni} options, {len(antisinergie)} antisynergies")
        print("Calling optimization server...")

        # Call remote optimization server
        server_response = call_optimization_server(
            capitale, unita_capitale, opzioni, antisinergie, penalty_strength=1.0
        )

        print("Server response received successfully!")

        # Write QUBO matrix to Calc
        scrivi_qubo_in_foglio_calc(
            server_response["qubo_matrix"],
            server_response["qubo_variables"],
            server_response["qubo_offset"],
            nome_foglio="QUBO_Matrix"
        )

        # Write results to Calc
        write_results_to_calc(server_response, opzioni_names)

        print("Results written to LibreOffice Calc successfully!")
        print(f"Best solution: {server_response['best_solution']['method']}")
        print(f"Best objective: {server_response['best_solution']['objective']:.6f}")

        # Show completion message
        ctx = uno.getComponentContext()
        smgr = ctx.ServiceManager
        desktop = smgr.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
        
        # Simple message (LibreOffice message box)
        try:
            toolkit = smgr.createInstance("com.sun.star.awt.Toolkit")
            msgbox = toolkit.createMessageBox(
                desktop.getCurrentFrame().getContainerWindow(),
                uno.Enum("com.sun.star.awt.MessageBoxType", "MESSAGEBOX"),
                uno.Enum("com.sun.star.awt.MessageBoxButtons", "BUTTONS_OK"),
                "Optimization Complete",
                f"Portfolio optimization completed successfully!\n\n"
                f"Best method: {server_response['best_solution']['method']}\n"
                f"Best objective: {server_response['best_solution']['objective']:.6f}\n"
                f"Total capital used: {server_response['best_solution']['total_capital']}\n\n"
                f"Check the 'Best_Solution' and 'Risultati' sheets for detailed results."
            )
            msgbox.execute()
        except Exception as e:
            print(f"Could not show message box: {e}")

    except Exception as e:
        error_msg = f"Error in portfolio optimization: {str(e)}"
        print(error_msg)
        
        # Try to show error message
        try:
            ctx = uno.getComponentContext()
            smgr = ctx.ServiceManager
            desktop = smgr.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
            toolkit = smgr.createInstance("com.sun.star.awt.Toolkit")
            msgbox = toolkit.createMessageBox(
                desktop.getCurrentFrame().getContainerWindow(),
                uno.Enum("com.sun.star.awt.MessageBoxType", "ERRORBOX"),
                uno.Enum("com.sun.star.awt.MessageBoxButtons", "BUTTONS_OK"),
                "Optimization Error",
                error_msg
            )
            msgbox.execute()
        except:
            pass  # If even error display fails, just print to console

# Alias for easier macro calling
def run_portfolio_optimization(*args):
    """Convenience function with shorter name"""
    return esegui_qubo_macro_server(*args)