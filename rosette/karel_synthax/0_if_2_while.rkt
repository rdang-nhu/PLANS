(define-synthax Block
  ([(Block previous_actions perceptions)
     (let ([new_actions (Action_Block previous_actions 10)])
            (let ([new_actions (While new_actions perceptions)])
                (let ([new_actions (Action_Block new_actions 10)])
                    (let ([new_actions (While new_actions perceptions)])
                          (Action_Block new_actions 10)))))]))