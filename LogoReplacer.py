from abc import ABC, abstractmethod


class LogoReplacer(ABC):
    @abstractmethod
    def build_model(self, filename):
        pass

    @abstractmethod
    def detect_object(self):
        pass

    @abstractmethod
    def insert_logo(self):
        pass


if __name__ == '__main__':
    print('Abstract class is ready to import')
