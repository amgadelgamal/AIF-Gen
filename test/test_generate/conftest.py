def mock_config_dict():
    config_dict = {
        'model_name': 'mock',
        'data': {
            'task_specs': [
                {
                    'num_samples': 5,
                    'alignment_task': {
                        'domain': {
                            'Component A': {
                                'seed_words': ['a_foo', 'a_bar', 'a_baz'],
                            },
                            'Component B': {
                                'seed_words': ['b_foo', 'b_bar', 'b_baz'],
                            },
                        },
                        'objective': 'Mock Objective 1',
                        'preference': 'Mock Preference 1',
                    },
                },
                {
                    'num_samples': 5,
                    'alignment_task': {
                        'domain': {
                            'Component A': {
                                'seed_words': ['a_foo', 'a_bar', 'a_baz'],
                            },
                            'Component B': {
                                'seed_words': ['b_foo', 'b_bar', 'b_baz'],
                            },
                        },
                        'objective': 'Mock Objective 2',
                        'preference': 'Mock Preference 2',
                    },
                },
            ]
        },
    }

    return config_dict
