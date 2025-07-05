python -m pytest -q --tb=short
C:\Users\Vatsal\AppData\Roaming\Python\Python313\site-packages\pytest_asyncio\plugin.py:208: PytestDeprecationWarning: The configuration option "asyncio_default_fixture_loop_scope" is unset.
The event loop scope for asynchronous fixtures will default to the fixture caching scope. Future versions of pytest-asyncio will default the loop scope for asynchronous fixtures to function scope. Set the default fixture loop scope explicitly in order to avoid unexpected behavior in the future. Valid fixture loop scopes are: "function", "class", "module", "package", "session"

  warnings.warn(PytestDeprecationWarning(_DEFAULT_FIXTURE_LOOP_SCOPE_UNSET))
================================================================ test session starts ================================================================
platform win32 -- Python 3.13.4, pytest-8.3.5, pluggy-1.6.0
rootdir: E:\friday
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.9.0, dash-3.0.2, asyncio-1.0.0, cov-6.2.1, mock-3.14.1
asyncio: mode=Mode.STRICT, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 372 items / 17 errors

====================================================================== ERRORS =======================================================================
____________________________________________ ERROR collecting tests/application/api/test_api_gateway.py _____________________________________________
C:\Python313\Lib\importlib\__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1331: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:935: in _load_unlocked
    ???
C:\Users\Vatsal\AppData\Roaming\Python\Python313\site-packages\_pytest\assertion\rewrite.py:185: in exec_module
    exec(co, module.__dict__)
tests\application\api\test_api_gateway.py:14: in <module>
    from src.application.api.main import create_app
src\application\api\main.py:14: in <module>
    from src.application.api.api_gateway import create_api_gateway
src\application\api\api_gateway.py:25: in <module>
    API_CONFIG = get_config("API_CONFIG")
src\infrastructure\config\__init__.py:271: in get_config
    raise KeyError(f"Configuration section '{section}' does not exist.")
E   KeyError: "Configuration section 'API_CONFIG' does not exist."
------------------------------------------------------------------ Captured stdout ------------------------------------------------------------------
Warning: Could not import from unified_config: cannot import name 'BROKER_CONFIG' from 'unified_config' (E:\friday\unified_config.py)
__________________________________________ ERROR collecting tests/application/test_knowledge_engine_api.py __________________________________________
ImportError while importing test module 'E:\friday\tests\application\test_knowledge_engine_api.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Python313\Lib\importlib\__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests\application\test_knowledge_engine_api.py:10: in <module>
    from src.application.api.knowledge_engine_api import router as knowledge_router
src\application\api\knowledge_engine_api.py:12: in <module>
    from src.infrastructure.auth.api_key_auth import api_key_auth
E   ImportError: cannot import name 'api_key_auth' from 'src.infrastructure.auth.api_key_auth' (E:\friday\src\infrastructure\auth\api_key_auth.py)
_____________________________________ ERROR collecting tests/data/acquisition/adapters/test_polygon_adapter.py ______________________________________
ImportError while importing test module 'E:\friday\tests\data\acquisition\adapters\test_polygon_adapter.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Python313\Lib\importlib\__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests\data\acquisition\adapters\test_polygon_adapter.py:13: in <module>
    from src.data.acquisition.adapters.polygon_adapter import PolygonAdapter
src\data\acquisition\adapters\polygon_adapter.py:15: in <module>
    from src.data.acquisition.adapters.financial_data_adapter import FinancialDataAdapter, FinancialDataType
src\data\acquisition\adapters\financial_data_adapter.py:19: in <module>
    from src.infrastructure.error.data_errors import DataConnectionError, DataValidationError
E   ModuleNotFoundError: No module named 'src.infrastructure.error.data_errors'
_______________________________________ ERROR collecting tests/data/acquisition/test_data_adapter_factory.py ________________________________________
ImportError while importing test module 'E:\friday\tests\data\acquisition\test_data_adapter_factory.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Python313\Lib\importlib\__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests\data\acquisition\test_data_adapter_factory.py:13: in <module>
    from src.data.acquisition.data_adapter_factory import DataAdapterFactory
src\data\acquisition\data_adapter_factory.py:11: in <module>
    from src.data.acquisition.adapters.financial_data_adapter import FinancialDataAdapter
src\data\acquisition\adapters\financial_data_adapter.py:19: in <module>
    from src.infrastructure.error.data_errors import DataConnectionError, DataValidationError
E   ModuleNotFoundError: No module named 'src.infrastructure.error.data_errors'
_____________________________________ ERROR collecting tests/data/alternative/test_alternative_data_service.py ______________________________________
ImportError while importing test module 'E:\friday\tests\data\alternative\test_alternative_data_service.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Python313\Lib\importlib\__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests\data\alternative\test_alternative_data_service.py:11: in <module>
    from src.data.alternative.alternative_data_service import AlternativeDataService
src\data\alternative\__init__.py:7: in <module>
    from src.data.alternative.news_sentiment_analyzer import NewsSentimentAnalyzer
src\data\alternative\news_sentiment_analyzer.py:20: in <module>
    from src.infrastructure.database.mongodb import (
E   ImportError: cannot import name 'find' from 'src.infrastructure.database.mongodb' (E:\friday\tests\communication\..\..\src\infrastructure\database\mongodb.py)
__________________________________________ ERROR collecting tests/data/alternative/test_error_handling.py ___________________________________________
ImportError while importing test module 'E:\friday\tests\data\alternative\test_error_handling.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Python313\Lib\importlib\__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests\data\alternative\test_error_handling.py:12: in <module>
    from src.data.alternative.error_handling import (
src\data\alternative\__init__.py:7: in <module>
    from src.data.alternative.news_sentiment_analyzer import NewsSentimentAnalyzer
src\data\alternative\news_sentiment_analyzer.py:20: in <module>
    from src.infrastructure.database.mongodb import (
E   ImportError: cannot import name 'find' from 'src.infrastructure.database.mongodb' (E:\friday\tests\communication\..\..\src\infrastructure\database\mongodb.py)
_______________________________________ ERROR collecting tests/infrastructure/database/test_database_setup.py _______________________________________
ImportError while importing test module 'E:\friday\tests\infrastructure\database\test_database_setup.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Python313\Lib\importlib\__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests\infrastructure\database\test_database_setup.py:18: in <module>
    from src.infrastructure.database.initialize_db import (
E   ImportError: cannot import name 'initialize_redis_structures' from 'src.infrastructure.database.initialize_db' (E:\friday\tests\communication\..\..\src\infrastructure\database\initialize_db.py)
____________________________________ ERROR collecting tests/integration/services/test_financial_data_service.py _____________________________________
ImportError while importing test module 'E:\friday\tests\integration\services\test_financial_data_service.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Python313\Lib\importlib\__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests\integration\services\test_financial_data_service.py:12: in <module>
    from src.integration.services.financial_data_service import FinancialDataService
src\integration\services\financial_data_service.py:13: in <module>
    from src.data.acquisition.adapters.financial_data_adapter import FinancialDataAdapter, FinancialDataType
src\data\acquisition\adapters\financial_data_adapter.py:19: in <module>
    from src.infrastructure.error.data_errors import DataConnectionError, DataValidationError
E   ModuleNotFoundError: No module named 'src.infrastructure.error.data_errors'
______________________________ ERROR collecting tests/orchestration/knowledge_engine/test_book_knowledge_extractor.py _______________________________
ImportError while importing test module 'E:\friday\tests\orchestration\knowledge_engine\test_book_knowledge_extractor.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Python313\Lib\importlib\__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests\orchestration\knowledge_engine\test_book_knowledge_extractor.py:18: in <module>
    from src.orchestration.knowledge_engine.book_knowledge_extractor import BookKnowledgeExtractor
src\orchestration\knowledge_engine\__init__.py:16: in <module>
    from orchestration.knowledge_engine.book_knowledge_extractor import BookKnowledgeExtractor
E   ModuleNotFoundError: No module named 'orchestration'
_______________________________ ERROR collecting tests/orchestration/knowledge_engine/test_knowledge_base_builder.py ________________________________
ImportError while importing test module 'E:\friday\tests\orchestration\knowledge_engine\test_knowledge_base_builder.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Python313\Lib\importlib\__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests\orchestration\knowledge_engine\test_knowledge_base_builder.py:18: in <module>
    from src.orchestration.knowledge_engine.knowledge_base_builder import KnowledgeBaseBuilder
src\orchestration\knowledge_engine\__init__.py:16: in <module>
    from orchestration.knowledge_engine.book_knowledge_extractor import BookKnowledgeExtractor
E   ModuleNotFoundError: No module named 'orchestration'
______________________________ ERROR collecting tests/orchestration/knowledge_engine/test_knowledge_engine_service.py _______________________________
ImportError while importing test module 'E:\friday\tests\orchestration\knowledge_engine\test_knowledge_engine_service.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Python313\Lib\importlib\__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests\orchestration\knowledge_engine\test_knowledge_engine_service.py:8: in <module>
    from src.orchestration.knowledge_engine.knowledge_engine_service import KnowledgeEngineService
src\orchestration\knowledge_engine\__init__.py:16: in <module>
    from orchestration.knowledge_engine.book_knowledge_extractor import BookKnowledgeExtractor
E   ModuleNotFoundError: No module named 'orchestration'
__________________________ ERROR collecting tests/orchestration/knowledge_engine/test_knowledge_extraction_integration.py ___________________________
ImportError while importing test module 'E:\friday\tests\orchestration\knowledge_engine\test_knowledge_extraction_integration.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Python313\Lib\importlib\__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests\orchestration\knowledge_engine\test_knowledge_extraction_integration.py:18: in <module>
    from src.orchestration.knowledge_engine.knowledge_extraction_integration import KnowledgeExtractionIntegration
src\orchestration\knowledge_engine\__init__.py:16: in <module>
    from orchestration.knowledge_engine.book_knowledge_extractor import BookKnowledgeExtractor
E   ModuleNotFoundError: No module named 'orchestration'
_____________________________________ ERROR collecting tests/orchestration/knowledge_engine/test_nlp_models.py ______________________________________
ImportError while importing test module 'E:\friday\tests\orchestration\knowledge_engine\test_nlp_models.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Python313\Lib\importlib\__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests\orchestration\knowledge_engine\test_nlp_models.py:17: in <module>
    from src.orchestration.knowledge_engine.book_knowledge_extractor import BookKnowledgeExtractor
src\orchestration\knowledge_engine\__init__.py:16: in <module>
    from orchestration.knowledge_engine.book_knowledge_extractor import BookKnowledgeExtractor
E   ModuleNotFoundError: No module named 'orchestration'
________________________________________ ERROR collecting tests/services/model/test_enhanced_model_backup.py ________________________________________
ImportError while importing test module 'E:\friday\tests\services\model\test_enhanced_model_backup.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Python313\Lib\importlib\__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests\services\model\test_enhanced_model_backup.py:22: in <module>
    from src.services.model.enhanced_model_backup import (
src\services\model\__init__.py:6: in <module>
    from src.services.model.model_registry import ModelRegistry
src\services\model\model_registry.py:24: in <module>
    from src.infrastructure.security.audit_logging import SecurityAuditLogger, log_model_created, log_model_updated, log_model_deleted, log_model_deployed, log_model_signed, log_model_signature_verified, log_model_signature_failed, log_model_encrypted, log_model_decrypted, log_model_access_denied, log_model_loaded, log_model_exported, log_model_decryption_failed
E   ImportError: cannot import name 'log_model_loaded' from 'src.infrastructure.security.audit_logging' (E:\friday\src\infrastructure\security\audit_logging.py)
__________________________________________ ERROR collecting tests/services/model/test_model_monitoring.py ___________________________________________
ImportError while importing test module 'E:\friday\tests\services\model\test_model_monitoring.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Python313\Lib\importlib\__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests\services\model\test_model_monitoring.py:20: in <module>
    from src.services.model.model_monitoring import (
src\services\model\__init__.py:6: in <module>
    from src.services.model.model_registry import ModelRegistry
src\services\model\model_registry.py:24: in <module>
    from src.infrastructure.security.audit_logging import SecurityAuditLogger, log_model_created, log_model_updated, log_model_deleted, log_model_deployed, log_model_signed, log_model_signature_verified, log_model_signature_failed, log_model_encrypted, log_model_decrypted, log_model_access_denied, log_model_loaded, log_model_exported, log_model_decryption_failed
E   ImportError: cannot import name 'log_model_loaded' from 'src.infrastructure.security.audit_logging' (E:\friday\src\infrastructure\security\audit_logging.py)
______________________________________ ERROR collecting tests/services/model/test_model_registry_extensions.py ______________________________________
ImportError while importing test module 'E:\friday\tests\services\model\test_model_registry_extensions.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Python313\Lib\importlib\__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests\services\model\test_model_registry_extensions.py:15: in <module>
    from src.services.model.model_registry import ModelRegistry
src\services\model\__init__.py:6: in <module>
    from src.services.model.model_registry import ModelRegistry
src\services\model\model_registry.py:24: in <module>
    from src.infrastructure.security.audit_logging import SecurityAuditLogger, log_model_created, log_model_updated, log_model_deleted, log_model_deployed, log_model_signed, log_model_signature_verified, log_model_signature_failed, log_model_encrypted, log_model_decrypted, log_model_access_denied, log_model_loaded, log_model_exported, log_model_decryption_failed
E   ImportError: cannot import name 'log_model_loaded' from 'src.infrastructure.security.audit_logging' (E:\friday\src\infrastructure\security\audit_logging.py)
_____________________________________ ERROR collecting tests/services/model/test_model_registry_integration.py ______________________________________
ImportError while importing test module 'E:\friday\tests\services\model\test_model_registry_integration.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Python313\Lib\importlib\__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests\services\model\test_model_registry_integration.py:21: in <module>
    from src.services.model.model_registry import ModelRegistry
src\services\model\__init__.py:6: in <module>
    from src.services.model.model_registry import ModelRegistry
src\services\model\model_registry.py:24: in <module>
    from src.infrastructure.security.audit_logging import SecurityAuditLogger, log_model_created, log_model_updated, log_model_deleted, log_model_deployed, log_model_signed, log_model_signature_verified, log_model_signature_failed, log_model_encrypted, log_model_decrypted, log_model_access_denied, log_model_loaded, log_model_exported, log_model_decryption_failed
E   ImportError: cannot import name 'log_model_loaded' from 'src.infrastructure.security.audit_logging' (E:\friday\src\infrastructure\security\audit_logging.py)
================================================================= warnings summary ==================================================================
tests\communication\test_bus.py:27
  E:\friday\tests\communication\test_bus.py:27: PytestCollectionWarning: cannot collect test class 'TestMessageHandler' because it has a __init__ constructor (from: tests/communication/test_bus.py)
    class TestMessageHandler(MessageHandler):

src\infrastructure\database\__init__.py:21
  E:\friday\tests\communication\..\..\src\infrastructure\database\__init__.py:21: MovedIn20Warning: The ``declarative_base()`` function is now available as sqlalchemy.orm.declarative_base(). (deprecated since: 2.0) (Background on SQLAlchemy 2.0 at: https://sqlalche.me/e/b8d9)
    Base = declarative_base()

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
============================================================== short test summary info ==============================================================
ERROR tests/application/api/test_api_gateway.py - KeyError: "Configuration section 'API_CONFIG' does not exist."
ERROR tests/application/test_knowledge_engine_api.py
ERROR tests/data/acquisition/adapters/test_polygon_adapter.py
ERROR tests/data/acquisition/test_data_adapter_factory.py
ERROR tests/data/alternative/test_alternative_data_service.py
ERROR tests/data/alternative/test_error_handling.py
ERROR tests/infrastructure/database/test_database_setup.py
ERROR tests/integration/services/test_financial_data_service.py
ERROR tests/orchestration/knowledge_engine/test_book_knowledge_extractor.py
ERROR tests/orchestration/knowledge_engine/test_knowledge_base_builder.py
ERROR tests/orchestration/knowledge_engine/test_knowledge_engine_service.py
ERROR tests/orchestration/knowledge_engine/test_knowledge_extraction_integration.py
ERROR tests/orchestration/knowledge_engine/test_nlp_models.py
ERROR tests/services/model/test_enhanced_model_backup.py
ERROR tests/services/model/test_model_monitoring.py
ERROR tests/services/model/test_model_registry_extensions.py
ERROR tests/services/model/test_model_registry_integration.py
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Interrupted: 17 errors during collection !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
========================================================== 2 warnings, 17 errors in 6.74s ===========================================================
