def is_connected(driver):
    try:
        driver.verify_connectivity()
        return True
    except Exception as e:
        print(f" There was an error : {e}")
