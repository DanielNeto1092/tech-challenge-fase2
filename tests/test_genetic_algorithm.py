from __future__ import annotations

import unittest

from diagnostico_saude_mulher.config.settings import AppSettings
from diagnostico_saude_mulher.genetic_algorithm.engine import ESPACOS_BUSCA, GeneticConfig, HyperparameterGeneticOptimizer, Individual


class GeneticAlgorithmTests(unittest.TestCase):
    def setUp(self) -> None:
        self.espaco_busca = ESPACOS_BUSCA["LogisticRegression"]
        self.optimizer = HyperparameterGeneticOptimizer(
            model_name="LogisticRegression",
            config=GeneticConfig(population_size=4, generations=2, mutation_rate=1.0, random_seed=7),
            settings=AppSettings(),
        )

    def test_populacao_inicial_respeita_espaco_busca(self) -> None:
        population = self.optimizer._initialize_population()
        self.assertEqual(len(population), 4)
        for individual in population:
            for nome, valor in individual.genes.items():
                self.assertIn(valor, self.espaco_busca[nome])

    def test_crossover_gera_filho_valido(self) -> None:
        parent1 = Individual(genes={nome: valores[0] for nome, valores in self.espaco_busca.items()})
        parent2 = Individual(genes={nome: valores[-1] for nome, valores in self.espaco_busca.items()})
        child = self.optimizer._crossover(parent1, parent2)
        for nome, valor in child.genes.items():
            self.assertIn(valor, {parent1.genes[nome], parent2.genes[nome]})

    def test_mutacao_mantem_dominio_dos_genes(self) -> None:
        original = Individual(genes={nome: valores[0] for nome, valores in self.espaco_busca.items()})
        mutated = self.optimizer._mutate(original)
        for nome, valor in mutated.genes.items():
            self.assertIn(valor, self.espaco_busca[nome])


if __name__ == "__main__":
    unittest.main()
